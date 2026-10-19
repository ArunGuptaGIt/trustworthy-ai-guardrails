#!/usr/bin/env python3
"""
Offline evaluation script for the guardrail system.

Usage:
    python evaluation/run_eval.py              # runs all evaluations, prints table
    python evaluation/run_eval.py --split test # test split only
    python evaluation/run_eval.py --out results.json

NEVER fabricate numbers. All metrics come from running this script.

Metrics computed:
    - Input action: Precision, Recall, F1 per action class; macro-averaged
    - PII detection: Precision, Recall, F1 (type-level)
    - Injection scoring: PR-AUC, F1 at chosen threshold
    - Groundedness: Precision, Recall, F1
    - Baseline comparison: same metrics for naive regex baseline

The baseline is implemented here and uses only:
    - A fixed keyword list for injection
    - A single email regex for PII
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

# Make sure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.guardrails.config import GuardrailConfig
from app.guardrails.groundedness import check_groundedness
from app.guardrails.injection import score_injection
from app.guardrails.input_pipeline import InputGuardrails
from app.guardrails.pii import detect_pii
from evaluation.datasets import (
    BASELINE_INJECTION_KEYWORDS,
    BASELINE_PII_PATTERNS,
    GROUNDEDNESS_DEV,
    GROUNDEDNESS_TEST,
    INJECTION_DEV,
    INJECTION_TEST,
    INPUT_DEV,
    INPUT_TEST,
    PII_DEV,
    PII_TEST,
)
from evaluation.harness import InMemoryAsyncRedis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    )
    return prec, rec, f1


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# ---------------------------------------------------------------------------
# Baseline implementation
# ---------------------------------------------------------------------------

class NaiveBaseline:
    """
    Naive regex-keyword baseline for comparison.

    Injection: keyword match (case-insensitive, exact phrase).
    PII: email regex only.
    Groundedness: always returns True (no groundedness check).
    Input action: if injection keyword found -> hard_block; if PII -> mask; else allow.
    """

    def classify_action(self, text: str) -> str:
        lower = text.lower()
        for kw in BASELINE_INJECTION_KEYWORDS:
            if kw in lower:
                return "hard_block"
        for pat in BASELINE_PII_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return "mask"
        return "allow"

    def detect_injection(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in BASELINE_INJECTION_KEYWORDS)

    def detect_pii_types(self, text: str) -> list[str]:
        found = []
        for pat in BASELINE_PII_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                found.append("pii:email")
        return found

    def check_groundedness(
        self, answer: str, context: list[str]
    ) -> bool:
        return True  # baseline doesn't check groundedness


baseline = NaiveBaseline()


# ---------------------------------------------------------------------------
# Input action evaluation
# ---------------------------------------------------------------------------

async def eval_input_actions(
    dataset: list[tuple[str, str, str, str]],
    label: str,
) -> dict[str, object]:
    cfg = GuardrailConfig()
    cfg.enabled = True
    cfg.mode = "enforce"
    redis = InMemoryAsyncRedis()
    guard = InputGuardrails(config=cfg, redis=redis)

    # Our system
    sys_results: list[tuple[str, str]] = []
    baseline_results: list[tuple[str, str]] = []
    cases: list[dict[str, object]] = []

    start = time.perf_counter()
    for index, (text, expected, category, note) in enumerate(dataset, start=1):
        d = await guard.check(text, "eval_session")
        sys_results.append((expected, d.action.value))
        baseline_results.append((expected, baseline.classify_action(text)))
        cases.append({
            "id": f"input_action_{index:03d}",
            "input": text,
            "reference_information": note,
            "expected_behavior": {"action": expected, "category": category},
            "actual_behavior": {"action": d.action.value, "findings": d.flags},
            "passed": d.action.value == expected,
        })
    elapsed = time.perf_counter() - start

    def compute_metrics(
        results: list[tuple[str, str]]
    ) -> dict[str, object]:
        # Per-class TP/FP/FN
        classes = sorted({exp for exp, _ in results})
        stats: dict[str, dict[str, int]] = {
            c: {"tp": 0, "fp": 0, "fn": 0} for c in classes
        }
        for exp, pred in results:
            if exp == pred:
                stats[exp]["tp"] += 1
            else:
                if pred in stats:
                    stats[pred]["fp"] += 1
                stats[exp]["fn"] += 1
        class_metrics: dict[str, dict[str, float]] = {}
        for c, s in stats.items():
            p, r, f = precision_recall_f1(s["tp"], s["fp"], s["fn"])
            class_metrics[c] = {"precision": p, "recall": r, "f1": f}
        # Macro average
        macro_p = sum(m["precision"] for m in class_metrics.values()) / max(len(class_metrics), 1)
        macro_r = sum(m["recall"] for m in class_metrics.values()) / max(len(class_metrics), 1)
        macro_f1 = sum(m["f1"] for m in class_metrics.values()) / max(len(class_metrics), 1)
        correct = sum(1 for e, p in results if e == p)
        return {
            "accuracy": correct / max(len(results), 1),
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_class": class_metrics,
        }

    return {
        "label": label,
        "n": len(dataset),
        "elapsed_s": round(elapsed, 3),
        "system": compute_metrics(sys_results),
        "baseline": compute_metrics(baseline_results),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# PII evaluation
# ---------------------------------------------------------------------------

def eval_pii(
    dataset: list[tuple[str, list[str], str]],
    label: str,
) -> dict[str, object]:
    sys_tp = sys_fp = sys_fn = 0
    base_tp = base_fp = base_fn = 0
    cases: list[dict[str, object]] = []

    for index, (text, expected_types, note) in enumerate(dataset, start=1):
        # System
        findings = detect_pii(text)
        found_types = list({f.type.value for f in findings})

        for et in expected_types:
            if any(ft == et for ft in found_types):
                sys_tp += 1
            else:
                sys_fn += 1
        for ft in found_types:
            if ft not in expected_types:
                sys_fp += 1

        # Baseline
        base_found = baseline.detect_pii_types(text)
        for et in expected_types:
            if et in base_found:
                base_tp += 1
            else:
                base_fn += 1
        for ft in base_found:
            if ft not in expected_types:
                base_fp += 1
        cases.append({
            "id": f"pii_{index:03d}",
            "input": text,
            "reference_information": note,
            "expected_behavior": {"finding_types": expected_types},
            "actual_behavior": {"finding_types": found_types},
            "passed": set(found_types) == set(expected_types),
        })

    s_p, s_r, s_f = precision_recall_f1(sys_tp, sys_fp, sys_fn)
    b_p, b_r, b_f = precision_recall_f1(base_tp, base_fp, base_fn)

    return {
        "label": label,
        "n": len(dataset),
        "system": {"precision": s_p, "recall": s_r, "f1": s_f},
        "baseline": {"precision": b_p, "recall": b_r, "f1": b_f},
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Injection evaluation
# ---------------------------------------------------------------------------

def eval_injection(
    dataset: list[tuple[str, bool, float, str]],
    label: str,
    threshold: float = 0.45,
) -> dict[str, object]:
    sys_tp = sys_fp = sys_fn = sys_tn = 0
    base_tp = base_fp = base_fn = base_tn = 0
    scores: list[tuple[float, bool]] = []
    cases: list[dict[str, object]] = []

    for index, (text, expected_fire, min_score, note) in enumerate(dataset, start=1):
        score, _findings = score_injection(text)
        scores.append((score, expected_fire))
        pred = score >= threshold
        if pred and expected_fire:
            sys_tp += 1
        elif pred and not expected_fire:
            sys_fp += 1
        elif not pred and expected_fire:
            sys_fn += 1
        else:
            sys_tn += 1

        base_pred = baseline.detect_injection(text)
        if base_pred and expected_fire:
            base_tp += 1
        elif base_pred and not expected_fire:
            base_fp += 1
        elif not base_pred and expected_fire:
            base_fn += 1
        else:
            base_tn += 1
        cases.append({
            "id": f"injection_{index:03d}",
            "input": text,
            "reference_information": note,
            "expected_behavior": {"flagged": expected_fire, "minimum_score": min_score},
            "actual_behavior": {"flagged": pred, "score": score},
            "passed": pred == expected_fire,
        })

    s_p, s_r, s_f = precision_recall_f1(sys_tp, sys_fp, sys_fn)
    b_p, b_r, b_f = precision_recall_f1(base_tp, base_fp, base_fn)

    # PR-AUC approximation (trapezoidal over sorted thresholds)
    sorted_scores = sorted(scores, key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    total_pos = sum(1 for _, pos in scores if pos)
    len(scores) - total_pos
    pr_points: list[tuple[float, float]] = []
    for _s, pos in sorted_scores:
        if pos:
            tp_cum += 1
        else:
            fp_cum += 1
        rec = tp_cum / max(total_pos, 1)
        prec = tp_cum / max(tp_cum + fp_cum, 1)
        pr_points.append((rec, prec))

    pr_auc = 0.0
    for i in range(1, len(pr_points)):
        r0, p0 = pr_points[i - 1]
        r1, p1 = pr_points[i]
        pr_auc += (r1 - r0) * (p0 + p1) / 2

    return {
        "label": label,
        "n": len(dataset),
        "threshold": threshold,
        "system": {
            "precision": s_p, "recall": s_r, "f1": s_f,
            "pr_auc": round(pr_auc, 4),
            "tp": sys_tp, "fp": sys_fp, "fn": sys_fn, "tn": sys_tn,
        },
        "baseline": {
            "precision": b_p, "recall": b_r, "f1": b_f,
            "tp": base_tp, "fp": base_fp, "fn": base_fn, "tn": base_tn,
        },
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Groundedness evaluation
# ---------------------------------------------------------------------------

def eval_groundedness(
    dataset: list[tuple[str, list[str], bool, str]],
    label: str,
    min_support: float = 0.50,
) -> dict[str, object]:
    from app.guardrails.types import OutputAction

    sys_tp = sys_fp = sys_fn = sys_tn = 0
    base_tp = base_fp = base_fn = 0
    cases: list[dict[str, object]] = []

    for index, (answer, context, expected_grounded, note) in enumerate(dataset, start=1):
        result = check_groundedness(answer, context, min_support=min_support)
        pred_grounded = result.action in (OutputAction.PASS, OutputAction.REDACT)
        if pred_grounded and expected_grounded:
            sys_tp += 1
        elif pred_grounded and not expected_grounded:
            sys_fp += 1
        elif not pred_grounded and expected_grounded:
            sys_fn += 1
        else:
            sys_tn += 1

        # Baseline: always True
        base_pred = True
        if base_pred and expected_grounded:
            base_tp += 1
        elif base_pred and not expected_grounded:
            base_fp += 1
        cases.append({
            "id": f"groundedness_{index:03d}",
            "input": answer,
            "reference_information": {"context": context, "note": note},
            "expected_behavior": {"grounded": expected_grounded},
            "actual_behavior": {
                "grounded": pred_grounded,
                "action": result.action.value,
                "supported_ratio": result.supported_ratio,
                "claims": [{"claim": claim, "score": score} for claim, score in result.per_claim],
            },
            "passed": pred_grounded == expected_grounded,
        })

    s_p, s_r, s_f = precision_recall_f1(sys_tp, sys_fp, sys_fn)
    b_p, b_r, b_f = precision_recall_f1(base_tp, base_fp, base_fn)

    return {
        "label": label,
        "n": len(dataset),
        "system": {
            "precision": s_p, "recall": s_r, "f1": s_f,
            "tp": sys_tp, "fp": sys_fp, "fn": sys_fn, "tn": sys_tn,
        },
        "baseline": {
            "precision": b_p, "recall": b_r, "f1": b_f,
        },
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(results: dict[str, object]) -> None:
    def row(label: str, sys_v: float, base_v: float, higher_better: bool = True) -> None:
        diff = sys_v - base_v
        marker = "+" if (diff >= 0) == higher_better else "-"
        print(
            f"  {label:<38} sys={fmt_pct(sys_v)}  base={fmt_pct(base_v)}"
            f"  delta={marker}{fmt_pct(abs(diff))}"
        )

    for section, data in results.items():
        d = data  # type: ignore[assignment]
        print(f"\n{'='*70}")
        print(f"  {section.upper()} ({d['label']}, n={d['n']})")
        print(f"{'='*70}")
        sys_m = d.get("system", {})
        base_m = d.get("baseline", {})
        for key in ("precision", "recall", "f1", "pr_auc"):
            sv = sys_m.get(key)
            bv = base_m.get(key)
            if sv is not None and bv is not None:
                row(key, float(sv), float(bv))
        if "accuracy" in sys_m:
            row("accuracy", float(sys_m["accuracy"]), float(base_m.get("accuracy", 0.0)))
        if "macro_f1" in sys_m:
            row("macro_f1", float(sys_m["macro_f1"]), float(base_m.get("macro_f1", 0.0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(split: str, out: str | None) -> None:
    print(f"\nEvaluating on split: {split}")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    all_results: dict[str, dict[str, object]] = {}

    if split == "dev":
        input_ds = INPUT_DEV
        pii_ds = PII_DEV
        inj_ds = INJECTION_DEV
        grnd_ds = GROUNDEDNESS_DEV
        lbl = "dev"
    else:
        input_ds = INPUT_TEST
        pii_ds = PII_TEST
        inj_ds = INJECTION_TEST
        grnd_ds = GROUNDEDNESS_TEST
        lbl = "test"

    print("\nRunning input action evaluation...")
    all_results["input_action"] = await eval_input_actions(input_ds, lbl)

    print("Running PII evaluation...")
    all_results["pii_detection"] = eval_pii(pii_ds, lbl)

    print("Running injection scoring evaluation...")
    all_results["injection"] = eval_injection(inj_ds, lbl)

    print("Running groundedness evaluation...")
    all_results["groundedness"] = eval_groundedness(grnd_ds, lbl)

    print_results(all_results)

    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to: {path}")

    # Summary line for report
    ia = all_results["input_action"]
    inj = all_results["injection"]
    pii = all_results["pii_detection"]
    grnd = all_results["groundedness"]

    print("\n" + "="*70)
    print("  SUMMARY (system vs baseline)")
    print("="*70)
    print(f"  Input action macro-F1: sys={fmt_pct(ia['system']['macro_f1'])}  "  # type: ignore[index]
          f"base={fmt_pct(ia['baseline']['macro_f1'])}")  # type: ignore[index]
    print(f"  Injection F1:          sys={fmt_pct(inj['system']['f1'])}  "  # type: ignore[index]
          f"base={fmt_pct(inj['baseline']['f1'])}")  # type: ignore[index]
    print(f"  PII F1:                sys={fmt_pct(pii['system']['f1'])}  "  # type: ignore[index]
          f"base={fmt_pct(pii['baseline']['f1'])}")  # type: ignore[index]
    print(f"  Groundedness F1:       sys={fmt_pct(grnd['system']['f1'])}  "  # type: ignore[index]
          f"base={fmt_pct(grnd['baseline']['f1'])}")  # type: ignore[index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Which dataset split to evaluate on (default: dev)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON results",
    )
    args = parser.parse_args()
    asyncio.run(main(args.split, args.out))
