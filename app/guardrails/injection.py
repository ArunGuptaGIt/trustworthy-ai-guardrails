"""
Prompt-injection detection using token-window rules (U2).

Detection approach:
- Each rule is a sequence of "windows": (verb_pattern, noun_pattern, max_distance_tokens).
- A rule fires when a verb token and a noun token appear within max_distance tokens
  of each other in the normalized token sequence.
- Rule weights are combined with noisy-OR: score = 1 - product(1 - w_i).

Mention-versus-use discount:
- If a match sits inside quoted text, or follows a framing phrase like
  "what does", "the phrase", "the document says", "example of",
  its weight is multiplied by the mention_discount factor (default 0.4).
- LIMITATION: an attacker who wraps an attack in quotes gets a lower score.
  This is explicitly documented as a trade-off.

Session risk:
- A decaying per-session risk score is maintained in Redis.
- The previous score is multiplied by decay_factor and added to the current turn's
  raw score to produce the effective score.
- This means repeated probing escalates the score over time.
- Key format: "guardrail:session_risk:{session_id}"

Category names (from instructor-style taxonomy):
  direct_injection, jailbreak, extraction, impersonation,
  context_manipulation, encoding_trick, exfiltration, multi_turn

LIMITATIONS (documented):
- Paraphrase attacks (semantically equivalent but different words) are missed.
- Multilingual attacks: only a few Romanized Nepali and Hindi phrases are included.
- Slow multi-turn build-up may escape the single-turn rules.
- Attacks embedded in images or files are out of scope.
- Quote-wrapping gives a ~60% discount (not a full block) which allows
  sophisticated attackers to probe with reduced scores.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.guardrails.types import Finding, FindingType

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class WindowRule:
    """
    A token-window rule.

    verb_re: regex matching the action/verb token
    noun_re: regex matching the target/noun token
    max_tokens: max distance in whitespace-split tokens between verb and noun
    weight: contribution weight (0-1) to the noisy-OR score
    category: injection category name
    """
    verb_re: re.Pattern[str]
    noun_re: re.Pattern[str]
    max_tokens: int
    weight: float
    category: str
    finding_type: FindingType


def _r(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Override / direct injection rules
_RULES: list[WindowRule] = [
    # "ignore/disregard/forget your instructions/rules/guidelines"
    WindowRule(
        verb_re=_r(r"^(ignore|disregard|forget|override|bypass|skip|cancel|disable|suppress|nullify)$"),
        noun_re=_r(r"^(instructions|rules|guidelines|directives|system|constraints|prompt|policy|policies)$"),
        max_tokens=6,
        weight=0.80,
        category="direct_injection",
        finding_type=FindingType.INJECTION_DIRECT,
    ),
    # "act as / you are now / pretend you are"
    WindowRule(
        verb_re=_r(r"^(act|pretend|roleplay|imagine|behave|respond)$"),
        noun_re=_r(r"^(as|if|like|that|you)$"),
        max_tokens=3,
        weight=0.70,
        category="jailbreak",
        finding_type=FindingType.INJECTION_JAILBREAK,
    ),
    # "you are now DAN / GPT / an AI without restrictions"
    WindowRule(
        verb_re=_r(r"^(you)$"),
        noun_re=_r(r"^(dan|gpt|unrestricted|jailbroken|free|unfiltered|uncensored)$"),
        max_tokens=5,
        weight=0.75,
        category="jailbreak",
        finding_type=FindingType.INJECTION_JAILBREAK,
    ),
    # "reveal / print / show / output the system prompt / instructions / api key"
    WindowRule(
        verb_re=_r(r"^(reveal|print|show|output|display|leak|return|repeat|expose|dump|exfiltrate|send|email)$"),
        noun_re=_r(r"^(prompt|instructions|system|secret|password|key|token|credentials|api|rules|config|configuration)$"),
        max_tokens=6,
        weight=0.85,
        category="extraction",
        finding_type=FindingType.INJECTION_EXTRACTION,
    ),
    # "ignore previous / above"
    WindowRule(
        verb_re=_r(r"^(ignore|disregard|forget|skip)$"),
        noun_re=_r(r"^(previous|above|prior|earlier|last|everything|all)$"),
        max_tokens=4,
        weight=0.75,
        category="direct_injection",
        finding_type=FindingType.INJECTION_DIRECT,
    ),
    # "new instruction / new prompt / new task"
    WindowRule(
        verb_re=_r(r"^(new|updated|different|alternative|real|actual|true)$"),
        noun_re=_r(r"^(instruction|instructions|prompt|task|goal|objective|role|persona|directive)$"),
        max_tokens=3,
        weight=0.65,
        category="direct_injection",
        finding_type=FindingType.INJECTION_DIRECT,
    ),
    # "impersonate / pretend to be admin / developer"
    WindowRule(
        verb_re=_r(r"^(impersonate|pose|masquerade|claim|pretend)$"),
        noun_re=_r(r"^(admin|administrator|developer|owner|root|staff|employee|system)$"),
        max_tokens=5,
        weight=0.70,
        category="impersonation",
        finding_type=FindingType.INJECTION_IMPERSONATION,
    ),
    # Context manipulation: "the document says / the context says to do X"
    WindowRule(
        verb_re=_r(r"^(says|states|tells|instructs|orders|commands|requires)$"),
        noun_re=_r(r"^(do|run|execute|perform|say|output|respond|ignore|reveal)$"),
        max_tokens=8,
        weight=0.55,
        category="context_manipulation",
        finding_type=FindingType.INJECTION_CONTEXT_MANIPULATION,
    ),
    # Encoding trick detection: base64, hex, rot13 framing
    WindowRule(
        verb_re=_r(r"^(decode|decrypt|deobfuscate|base64|hex|rot13|evaluate|execute|eval|run)$"),
        noun_re=_r(r"^(this|the|following|encoded|below|payload|string|text|message|instruction)$"),
        max_tokens=5,
        weight=0.80,
        category="encoding_trick",
        finding_type=FindingType.INJECTION_ENCODING_TRICK,
    ),
    # Exfiltration: "send all emails / share all bookings / show all records"
    WindowRule(
        verb_re=_r(r"^(send|share|list|show|dump|export|give|provide|email|forward)$"),
        noun_re=_r(r"^(all|every|each|any|other|others|everyone|people)$"),
        max_tokens=4,
        weight=0.60,
        category="exfiltration",
        finding_type=FindingType.INJECTION_EXFILTRATION,
    ),
]

# Romanized Nepali / Hindi injection phrases (direct patterns, not window rules)
_ROMANIZED_PATTERNS: list[tuple[re.Pattern[str], str, FindingType, float]] = [
    (
        _r(r"\b(niyam\s+bhul|nirdesh\s+bhul|sabai\s+niyam|paila\s+nirdesh)\b"),
        "direct_injection", FindingType.INJECTION_DIRECT, 0.70,
    ),
    (
        _r(r"\b(apni\s+pahchan|batao\s+secret|mujhe\s+batao)\b"),
        "extraction", FindingType.INJECTION_EXTRACTION, 0.65,
    ),
    (
        _r(r"\b(system\s+prompt\s+batao|instructions\s+dikhao)\b"),
        "extraction", FindingType.INJECTION_EXTRACTION, 0.75,
    ),
]

# Framing patterns that indicate mention-rather-than-use context
_MENTION_FRAMES: list[re.Pattern[str]] = [
    _r(r"\b(what\s+does|what\s+is|the\s+phrase|the\s+document\s+says|example\s+of|"
       r"definition\s+of|explain\s+what|what\s+would|how\s+would|suppose|hypothetically|"
       r"if\s+someone|as\s+an\s+example|for\s+instance|suppose\s+that)\b"),
]

# Quote detection: check if match is inside "..." or '...'
_QUOTED = re.compile(r'"[^"]{0,200}"' + r"|'[^']{0,200}'")


def _is_in_mention_context(text: str, start: int, end: int) -> bool:
    """Return True if the match at [start, end) appears inside a mention/quote frame."""
    window_start = max(0, start - 100)
    preceding = text[window_start:start]
    # Check for framing phrases
    for frame in _MENTION_FRAMES:
        if frame.search(preceding):
            return True
    # Check if enclosed in quotes
    for qm in _QUOTED.finditer(text):
        if qm.start() <= start and end <= qm.end():
            return True
    return False


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    # Split on whitespace and punctuation, keep tokens
    tokens = re.split(r"[\s,;:!?.\"'()\[\]{}]+", text.lower())
    return [t for t in tokens if t]


def _check_window_rule(tokens: list[str], rule: WindowRule) -> tuple[bool, int, int]:
    """
    Return (fired, verb_idx, noun_idx) for the first occurrence of the rule.
    """
    for i, tok in enumerate(tokens):
        if rule.verb_re.match(tok):
            # Look within max_tokens
            for j in range(i + 1, min(len(tokens), i + rule.max_tokens + 1)):
                if rule.noun_re.match(tokens[j]):
                    return True, i, j
    return False, -1, -1


def _approx_char_span(text: str, tokens: list[str], verb_idx: int, noun_idx: int) -> tuple[int, int]:
    """Approximate character span from token indices (best-effort)."""
    # Rebuild by counting tokens in order
    lower_text = text.lower()
    pos = 0
    start_char = 0
    end_char = len(text)
    for k, tok in enumerate(tokens):
        found = lower_text.find(tok, pos)
        if found == -1:
            continue
        if k == verb_idx:
            start_char = found
        if k == noun_idx:
            end_char = found + len(tok)
        pos = found + len(tok)
    return start_char, end_char


def score_injection(
    text: str,
    session_risk: float = 0.0,
    mention_discount: float = 0.4,
) -> tuple[float, list[Finding]]:
    """
    Score the text for prompt injection.

    Returns:
        (effective_score, findings)

    effective_score includes session_risk contribution:
        noisy_or_score_including_previous_session_risk
    """
    tokens = _tokenize(text)
    fired_weights: list[float] = []
    findings: list[Finding] = []

    # Check window rules
    for rule in _RULES:
        fired, vi, ni = _check_window_rule(tokens, rule)
        if fired:
            span = _approx_char_span(text, tokens, vi, ni)
            in_mention = _is_in_mention_context(text, span[0], span[1])
            w = rule.weight * (mention_discount if in_mention else 1.0)
            fired_weights.append(w)
            findings.append(
                Finding(
                    type=rule.finding_type,
                    detail=f"rule:{rule.category} mention={in_mention}",
                    span=span,
                    weight=w,
                )
            )

    # Check romanized patterns
    for pat, cat, ft, w in _ROMANIZED_PATTERNS:
        m = pat.search(text)
        if m:
            in_mention = _is_in_mention_context(text, m.start(), m.end())
            effective_w = w * (mention_discount if in_mention else 1.0)
            fired_weights.append(effective_w)
            findings.append(
                Finding(
                    type=ft,
                    detail=f"romanized:{cat} mention={in_mention}",
                    span=(m.start(), m.end()),
                    weight=effective_w,
                )
            )

    # Noisy-OR combination
    raw_score: float = 1.0 - math.prod(1.0 - w for w in fired_weights) if fired_weights else 0.0

    # Add session risk (noisy-OR with previous session risk)
    effective_score: float = 1.0 - (1.0 - raw_score) * (1.0 - session_risk)

    return effective_score, findings


# Redis key helpers for session risk
def session_risk_key(session_id: str) -> str:
    return f"guardrail:session_risk:{session_id}"


async def get_session_risk(redis: Any, session_id: str) -> float:
    """Read the current session risk score from Redis."""
    try:
        val = await redis.get(session_risk_key(session_id))
        if val is None:
            return 0.0
        return float(val)
    except Exception:
        return 0.0


async def update_session_risk(
    redis: Any,
    session_id: str,
    new_turn_score: float,
    decay: float = 0.7,
    ttl_seconds: int = 3600,
) -> float:
    """
    Update session risk: decayed_previous + noisy-OR with new score.
    Returns the updated score.
    """
    try:
        prev = await get_session_risk(redis, session_id)
        decayed = prev * decay
        updated = 1.0 - (1.0 - decayed) * (1.0 - new_turn_score)
        await redis.set(session_risk_key(session_id), str(updated), ex=ttl_seconds)
        return updated
    except Exception:
        return new_turn_score
