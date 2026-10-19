"""
Output PII and secret detection with provenance classification (U8).

Every PII or secret hit in an answer is classified as:
  session_own    : matches the current user's booking state or input message
  context_sourced: appears in a retrieved chunk
  unknown        : neither (possibly hallucinated or memorized)

Policy:
  session_own    : allowed only in booking replies (redact in RAG answers)
  context_sourced: redacted unless allow_pii_from_context is True
  unknown        : always redacted and flagged

System-prompt leakage detection:
  1. Canary token check: if the canary token appears verbatim in the answer
  2. N-gram overlap: if 8-gram overlap fraction > threshold with system prompt

Actions: REDACT (PII), REFUSE (secrets, canary, prompt-overlap leak).
"""

from __future__ import annotations

import logging

from app.guardrails.pii import detect_pii, mask_pii
from app.guardrails.types import Finding, FindingType, OutputAction, PIIProvenance

logger = logging.getLogger("app.guardrails.output_sensitive")


def _get_ngrams(text: str, n: int) -> set[str]:
    """Return the set of n-grams (over whitespace-split tokens)."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def check_canary_leak(
    answer: str,
    canary_token: str,
    system_prompt: str = "",
    ngram_size: int = 8,
    overlap_threshold: float = 0.3,
) -> tuple[bool, str]:
    """
    Check for system-prompt leakage.

    Returns (leaked, detail).
    """
    # 1. Verbatim canary token
    if canary_token and canary_token in answer:
        return True, f"canary_token_found:{canary_token[:8]}..."

    # 2. N-gram overlap with system prompt
    if system_prompt:
        prompt_ngrams = _get_ngrams(system_prompt, ngram_size)
        answer_ngrams = _get_ngrams(answer, ngram_size)
        if prompt_ngrams:
            overlap = len(prompt_ngrams & answer_ngrams) / len(prompt_ngrams)
            if overlap >= overlap_threshold:
                return True, f"ngram_overlap={overlap:.2f}"

    return False, ""


def classify_pii_provenance(
    value_fragment: str,
    session_context: str,
    retrieved_chunks: list[str],
) -> PIIProvenance:
    """
    Classify where a PII value came from.

    value_fragment: a short fragment of the masked PII (enough to search)
    session_context: concatenation of session state (user messages, booking state)
    retrieved_chunks: the context chunks used in this turn
    """
    # Check session context first (own session data)
    if value_fragment and len(value_fragment) > 3 and value_fragment.lower() in session_context.lower():
        return PIIProvenance.SESSION_OWN

    # Check retrieved chunks
    for chunk in retrieved_chunks:
        if value_fragment and len(value_fragment) > 3 and value_fragment.lower() in chunk.lower():
            return PIIProvenance.CONTEXT_SOURCED

    return PIIProvenance.UNKNOWN


def check_output_sensitive(
    answer: str,
    retrieved_chunks: list[str],
    session_context: str = "",
    allow_pii_from_context: bool = False,
    is_booking_reply: bool = False,
    canary_token: str = "",
    system_prompt: str = "",
    ngram_size: int = 8,
    overlap_threshold: float = 0.3,
) -> tuple[str, list[Finding], OutputAction]:
    """
    Check the output answer for PII leaks, secret leaks, and prompt leakage.

    Returns:
        (final_text, findings, action)

    action is REFUSE for canary/prompt leakage, REDACT for PII.
    """
    findings: list[Finding] = []
    action = OutputAction.PASS

    # 1. Check for system-prompt leakage (REFUSE)
    leaked, detail = check_canary_leak(answer, canary_token, system_prompt, ngram_size, overlap_threshold)
    if leaked:
        findings.append(
            Finding(
                type=FindingType.OUTPUT_CANARY_LEAK if "canary" in detail else FindingType.OUTPUT_PROMPT_OVERLAP,
                detail=detail,
            )
        )
        logger.warning("System-prompt leakage detected: %s", detail)
        return answer, findings, OutputAction.REFUSE

    # 2. PII detection in the answer
    pii_findings = detect_pii(answer)
    if not pii_findings:
        return answer, findings, action

    final_text = answer
    needs_redaction = False

    for pf in pii_findings:
        # Get a short fragment for provenance lookup
        if pf.span is not None:
            fragment = answer[pf.span[0] : pf.span[1]][:40]
        else:
            fragment = ""

        provenance = classify_pii_provenance(fragment, session_context, retrieved_chunks)
        pf.provenance = provenance

        # Policy decision
        if provenance == PIIProvenance.SESSION_OWN:
            if not is_booking_reply:
                # Own PII in a RAG answer: redact
                needs_redaction = True
                findings.append(
                    Finding(
                        type=pf.type,
                        detail="session_own PII in RAG answer",
                        span=pf.span,
                        masked_preview=pf.masked_preview,
                        provenance=provenance,
                    )
                )
            # else: booking reply is allowed to contain own session data
        elif provenance == PIIProvenance.CONTEXT_SOURCED:
            if not allow_pii_from_context:
                needs_redaction = True
                findings.append(
                    Finding(
                        type=pf.type,
                        detail="context_sourced PII, allow_pii_from_context=false",
                        span=pf.span,
                        masked_preview=pf.masked_preview,
                        provenance=provenance,
                    )
                )
        else:  # UNKNOWN
            needs_redaction = True
            findings.append(
                Finding(
                    type=pf.type,
                    detail="unknown provenance PII",
                    span=pf.span,
                    masked_preview=pf.masked_preview,
                    provenance=provenance,
                )
            )
            # Secrets with unknown provenance -> REFUSE
            if pf.type == FindingType.PII_SECRET:
                action = OutputAction.REFUSE

    if needs_redaction and action == OutputAction.PASS:
        action = OutputAction.REDACT
        final_text = mask_pii(answer, findings)

    return final_text, findings, action
