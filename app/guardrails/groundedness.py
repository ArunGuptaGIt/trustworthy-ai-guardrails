"""
Groundedness validation (U7).

Checks whether the LLM's answer is supported by the retrieved context chunks.

Approach:
1. Split answer into claims (sentences, with abbreviation-aware splitter).
2. For each non-meta claim, score it against all context sentences using
   IDF-weighted content-token recall.
3. Additional checks: negation polarity mismatch, quantifier mismatch,
   numeric/date/amount exact comparison.
4. Canonicalize numbers, amounts (Rs./NPR), dates, and times before comparing.
5. Per-claim verdict: SUPPORTED, UNSUPPORTED, or META (skip).
6. Overall decision based on ratio of supported claims.

Decision thresholds (from GuardrailConfig):
  ratio >= groundedness_min_claim_support (default 0.50) AND NOT worse -> PASS
  ratio >= groundedness_redact_threshold (default 0.70)                -> REDACT
  else                                                                  -> REGENERATE (then FALLBACK)

For professional_advice topic: use groundedness_professional_threshold (0.85).

ClaimVerifier protocol: defined as a typing Protocol so an NLI model could
be swapped in without changing the pipeline.  Only LexicalVerifier is
implemented here.

LIMITATIONS (documented):
- Semantic paraphrase: synonyms and restructured sentences may score as
  UNSUPPORTED even when the meaning is present in context.
- Contradiction without negation words is missed.
- Multi-hop reasoning (claim requires combining two chunks) scores as UNSUPPORTED.
- Numbers written as words ("fifteen hundred") are partially handled
  via IDF overlap but not canonicalized.
- Unsupported claims built from words that happen to appear in context
  may score as SUPPORTED (false negative).
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from app.guardrails.types import Finding, FindingType, OutputAction

# ---------------------------------------------------------------------------
# Sentence splitter (handles Rs., e.g., decimals, Mr./Ms./Dr.)
# ---------------------------------------------------------------------------

_ABBREV = re.compile(
    r"\b(Rs|NPR|Mr|Mrs|Ms|Dr|Prof|St|vs|etc|i\.e|e\.g|approx|est|ref|no|vol|fig|ch|sec|dept|govt|univ|corp)\.",
    re.IGNORECASE,
)
_SENT_BOUNDARY = re.compile(r"(?<![A-Z])(?<!\d)\.(?!\d)(?!\w)|[!?]")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, preserving abbreviations like 'Rs.', 'e.g.', decimals.
    """
    # Temporarily replace abbreviation dots
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    sentences: list[str] = []
    for part in parts:
        # Restore protected dots
        restored = part.replace("\x00", ".")
        stripped = restored.strip()
        if stripped:
            sentences.append(stripped)
    return sentences if sentences else [text.strip()]


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def _normalize_amount(text: str) -> str:
    """Normalize Rs. 1,500 / NPR 1500 / 1500 rupees -> '1500'."""
    text = re.sub(r"Rs\.?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"NPR\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r",", "", text)
    text = re.sub(r"\s*(rupees?|taka)\b", "", text, flags=re.IGNORECASE)
    return text.strip()


_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _normalize_date(text: str) -> str:
    """Normalize date variants to YYYY-MM-DD."""
    # Already ISO
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text.strip())
    if m:
        return text.strip()
    # DD Mon YYYY or D Mon YYYY
    _patt_dmy = r"(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})"
    m2 = re.match(_patt_dmy, text.strip(), re.IGNORECASE)
    if m2:
        day, mon, year = m2.group(1), m2.group(2).lower()[:3], m2.group(3)
        return f"{year}-{_MONTH_MAP[mon]}-{day.zfill(2)}"
    # Mon DD, YYYY
    _patt_mdy = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2}),?\s+(\d{4})"
    m3 = re.match(_patt_mdy, text.strip(), re.IGNORECASE)
    if m3:
        mon, day, year = m3.group(1).lower()[:3], m3.group(2), m3.group(3)
        return f"{year}-{_MONTH_MAP[mon]}-{day.zfill(2)}"
    return text.strip()


def _extract_numbers(text: str) -> list[str]:
    """Extract all standalone numbers (integers and decimals) from text.
    Commas are stripped so '1,500' and '1500' compare as equal."""
    raw = re.findall(r"\b\d[\d,\.]*\d|\b\d\b", text)
    return [r.replace(",", "") for r in raw]


def _extract_amounts(text: str) -> list[str]:
    """Extract amount values after Rs./NPR canonicalization."""
    _amt_re = r"(?:Rs\.?\s*|NPR\s*)[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:rupees?|taka)\b"
    parts = re.findall(_amt_re, text, re.IGNORECASE)
    return [_normalize_amount(p) for p in parts]


def _extract_dates(text: str) -> list[str]:
    """Extract and normalize date-like strings."""
    patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}",
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}",
    ]
    dates: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            dates.append(_normalize_date(m.group(0)))
    return dates


# ---------------------------------------------------------------------------
# IDF computation (simple, over context sentences)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "and", "or", "but",
    "not", "no", "this", "that", "these", "those", "it", "its", "we", "i",
    "you", "he", "she", "they", "them", "our", "your", "their", "from",
    "as", "will", "can", "do", "does", "did", "have", "has", "had", "would",
    "could", "should", "may", "might", "shall", "which", "who", "when",
    "where", "how", "what", "all", "any", "each", "more", "also", "than",
    "about", "up", "out", "into", "if", "so", "then",
}


def _tokenize_content(text: str) -> list[str]:
    """Lowercase alphanum tokens, excluding stop words.
    Commas are stripped from numbers first so '1,500' tokenizes as '1500'."""
    # Strip commas inside numeric sequences
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _compute_idf(corpus: list[str]) -> dict[str, float]:
    """Very simple IDF over the given corpus of strings."""
    import math
    n = max(len(corpus), 1)
    df: dict[str, int] = {}
    for doc in corpus:
        for tok in set(_tokenize_content(doc)):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(n / cnt) for tok, cnt in df.items()}


# ---------------------------------------------------------------------------
# Negation and quantifier checks
# ---------------------------------------------------------------------------

_REDACT_NOTE = (
    "\n\n(Note: some details were removed because they could not be verified"
    " against the provided documents.)"
)

_NEGATIONS = re.compile(
    r"\b(not|never|no|none|neither|nor|cannot|can't|isn't|aren't|wasn't|weren't|don't|doesn't|didn't)\b",
    re.IGNORECASE,
)
_QUANTIFIERS = re.compile(r"\b(all|every|always|only|exclusively|solely|entire|complete)\b", re.IGNORECASE)


def _has_negation(text: str) -> bool:
    return bool(_NEGATIONS.search(text))


def _has_strong_quantifier(text: str) -> bool:
    return bool(_QUANTIFIERS.search(text))


# ---------------------------------------------------------------------------
# Meta-sentence detection (skip from groundedness check)
# ---------------------------------------------------------------------------

_META_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(the documents? do not contain|not found in the|unable to find|"
               r"no information|based on the (provided|retrieved|given)|"
               r"according to the (documents?|context|information)|"
               r"the (documents?|context) (says?|states?|indicates?)|"
               r"hello|hi,?|thank you|sure|of course|happy to help)\b", re.IGNORECASE),
]


def _is_meta_sentence(sentence: str) -> bool:
    for p in _META_PATTERNS:
        if p.search(sentence):
            return True
    return False


# ---------------------------------------------------------------------------
# ClaimVerifier protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ClaimVerifier(Protocol):
    """
    Protocol for a claim verifier.

    LexicalVerifier is the default implementation.
    An NLI model could be swapped in by implementing this protocol.
    """

    def verify(self, claim: str, context_sentences: list[str], idf: dict[str, float]) -> float:
        """Return a support score in [0, 1]."""
        ...


class LexicalVerifier:
    """
    IDF-weighted content-token recall verifier.

    For each claim sentence:
    - Tokenize claim and all context sentences
    - Score = fraction of claim content tokens (weighted by IDF) found in any context sentence
    - Additional checks: number/amount/date exact match, negation polarity
    """

    def verify(self, claim: str, context_sentences: list[str], idf: dict[str, float]) -> float:
        claim_tokens = set(_tokenize_content(claim))
        if not claim_tokens:
            return 1.0  # empty/meta claim passes

        # Combine all context into one searchable pool
        context_all = " ".join(context_sentences)
        context_tokens = set(_tokenize_content(context_all))

        # IDF-weighted recall.
        # When all IDF values are 0 (single-doc corpus, log(1/1)=0), use a
        # uniform weight for every token - including ones not in the dict -
        # so claim-specific synonyms like "NPR" vs "Rs." are not penalised.
        all_zero_idf = all(v == 0.0 for v in idf.values()) if idf else True
        _uniform = 0.1
        if all_zero_idf:
            raw_weights = dict.fromkeys(claim_tokens, _uniform)
        else:
            raw_weights = {t: max(idf.get(t, 0.5), 0.1) for t in claim_tokens}
        total_weight = sum(raw_weights.values())
        matched_weight = sum(raw_weights[t] for t in claim_tokens if t in context_tokens)
        base_score = matched_weight / total_weight if total_weight > 0 else 1.0

        # Exact numeric checks: if claim mentions numbers not in context, penalize
        claim_numbers = set(_extract_numbers(claim))
        claim_amounts = set(_extract_amounts(claim))
        claim_dates = set(_extract_dates(claim))
        context_numbers = set(_extract_numbers(context_all))
        context_amounts = set(_extract_amounts(context_all))
        context_dates = set(_extract_dates(context_all))

        # Unmatched numbers in the claim are strong evidence of ungroundedness
        unmatched_numbers = claim_numbers - context_numbers
        unmatched_amounts = claim_amounts - context_amounts
        unmatched_dates = claim_dates - context_dates

        if unmatched_numbers or unmatched_amounts or unmatched_dates:
            # Penalize proportionally to how many are unmatched
            n_exact = len(unmatched_numbers) + len(unmatched_amounts) + len(unmatched_dates)
            base_score *= max(0.0, 1.0 - 0.25 * n_exact)

        # Negation polarity: claim is negated but context is not (or vice versa)
        claim_negated = _has_negation(claim)
        context_negated = _has_negation(context_all)
        if claim_negated != context_negated and base_score > 0.5:
            base_score *= 0.5  # likely polarity mismatch

        # Quantifier mismatch: claim says "all" but context doesn't support that
        if _has_strong_quantifier(claim) and not _has_strong_quantifier(context_all):
            base_score *= 0.7

        return min(1.0, max(0.0, base_score))


# ---------------------------------------------------------------------------
# Main groundedness check
# ---------------------------------------------------------------------------

class GroundednessResult:
    def __init__(
        self,
        action: OutputAction,
        final_text: str,
        supported_ratio: float,
        findings: list[Finding],
        per_claim: list[tuple[str, float]],
    ) -> None:
        self.action = action
        self.final_text = final_text
        self.supported_ratio = supported_ratio
        self.findings = findings
        self.per_claim = per_claim  # (sentence, score) pairs


def check_groundedness(
    answer: str,
    retrieved_chunks: list[str],
    min_support: float = 0.50,
    redact_threshold: float = 0.70,
    verifier: ClaimVerifier | None = None,
) -> GroundednessResult:
    """
    Check whether the answer is grounded in the retrieved context.

    Args:
        answer: the LLM-generated answer text
        retrieved_chunks: list of chunk text strings
        min_support: minimum fraction of supported claims for PASS
        redact_threshold: if ratio falls below this, try REGENERATE instead of REDACT
        verifier: ClaimVerifier instance (default: LexicalVerifier)

    Returns:
        GroundednessResult with action, final_text (possibly redacted), findings
    """
    if verifier is None:
        verifier = LexicalVerifier()

    findings: list[Finding] = []

    if not retrieved_chunks:
        # No context: all claims are unsupported
        findings.append(Finding(type=FindingType.OUTPUT_UNGROUNDED, detail="no retrieved chunks"))
        return GroundednessResult(
            action=OutputAction.REGENERATE,
            final_text=answer,
            supported_ratio=0.0,
            findings=findings,
            per_claim=[],
        )

    # Build IDF over context sentences
    all_context_sents: list[str] = []
    for chunk in retrieved_chunks:
        all_context_sents.extend(split_sentences(chunk))
    idf = _compute_idf(all_context_sents)

    # Split answer into claims
    claim_sentences = split_sentences(answer)

    claim_scores: list[tuple[str, float]] = []
    for sent in claim_sentences:
        if _is_meta_sentence(sent):
            claim_scores.append((sent, 1.0))  # meta/boilerplate passes
            continue
        score = verifier.verify(sent, all_context_sents, idf)
        claim_scores.append((sent, score))

    # Compute support ratio (non-meta claims only)
    non_meta = [(s, sc) for s, sc in claim_scores if not _is_meta_sentence(s)]
    if not non_meta:
        return GroundednessResult(
            action=OutputAction.PASS,
            final_text=answer,
            supported_ratio=1.0,
            findings=[],
            per_claim=claim_scores,
        )

    supported = sum(1 for _, sc in non_meta if sc >= 0.5)
    ratio = supported / len(non_meta)

    # Decide action
    if ratio >= min_support:
        # Redact unsupported sentences (those with score < 0.5)
        unsupported_sents = {s for s, sc in non_meta if sc < 0.5}
        if unsupported_sents:
            kept = [s for s in claim_sentences if s not in unsupported_sents or _is_meta_sentence(s)]
            redacted_text = " ".join(kept)
            if redacted_text.strip():
                redacted_text += _REDACT_NOTE
            for sent in unsupported_sents:
                findings.append(Finding(type=FindingType.OUTPUT_UNGROUNDED, detail=f"unsupported: {sent[:80]}"))
            action = OutputAction.REDACT
            final_text = redacted_text
        else:
            action = OutputAction.PASS
            final_text = answer
    elif ratio >= redact_threshold:
        # Try to redact - enough supported to be useful
        unsupported_sents = {s for s, sc in non_meta if sc < 0.5}
        kept = [s for s in claim_sentences if s not in unsupported_sents or _is_meta_sentence(s)]
        redacted_text = " ".join(kept)
        if redacted_text.strip():
            redacted_text += _REDACT_NOTE
        for sent in unsupported_sents:
            findings.append(Finding(type=FindingType.OUTPUT_UNGROUNDED, detail=f"unsupported: {sent[:80]}"))
        action = OutputAction.REDACT
        final_text = redacted_text
    else:
        for _, sc in non_meta:
            if sc < 0.5:
                findings.append(Finding(type=FindingType.OUTPUT_UNGROUNDED, detail=f"support_ratio={ratio:.2f}"))
                break
        action = OutputAction.REGENERATE
        final_text = answer

    return GroundednessResult(
        action=action,
        final_text=final_text,
        supported_ratio=ratio,
        findings=findings,
        per_claim=claim_scores,
    )
