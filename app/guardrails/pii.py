"""
PII detection and masking (U4).

Detectors (in order of application):
1. Secrets / API keys (entropy + pattern): MASK with notice
2. Credit/debit cards with Luhn check: MASK with notice
3. Email addresses (multiple obfuscation forms): MASK
4. Nepali mobile numbers (96x/97x/98x with optional +977 prefix): MASK
5. Kathmandu-style landlines (01-XXXXXXX): MASK
6. Generic international phone numbers: MASK
7. Nepal citizenship certificate (context-gated): MASK
8. Nepal PAN (context-gated): MASK
9. National ID / passport / bank account (context-gated): MASK
10. IPv4 addresses: MASK

Masking uses typed placeholders so re-scanning masked text finds nothing.
Findings store type, span, and a masked preview only - never the raw value.

LIMITATIONS (documented in this module and in the report):
- No NER: names and addresses are not detected.
- Spelled-out digits ("nine eight four...") are not detected.
- Numbers split across sentences are not detected.
- Context-gated types (PAN, citizenship, etc.) require a nearby keyword;
  they will be missed if the keyword is absent.
- Nepal ID formats used here are NOT verified official specs; they are
  reasonable approximations. Stated as an assumption.
- Booking email tokenization: during an active booking session, the email is
  extracted locally and replaced with [EMAIL_1] before any LLM call.
  The real email is passed out-of-band to the booking service.
"""
from __future__ import annotations

import math
import re

from app.guardrails.types import Finding, FindingType

# ---------------------------------------------------------------------------
# Placeholder tokens (idempotent: re-scanning finds nothing new)
# ---------------------------------------------------------------------------
_EMAIL_PH = "[EMAIL]"
_PHONE_PH = "[PHONE]"
_CARD_PH = "[CARD]"
_ID_PH = "[ID_NUMBER]"
_SECRET_PH = "[SECRET]"
_IP_PH = "[IP_ADDRESS]"
_EMAIL_BOOK_PH = "[EMAIL_1]"  # used during booking tokenization

_PLACEHOLDERS = {_EMAIL_PH, _PHONE_PH, _CARD_PH, _ID_PH, _SECRET_PH, _IP_PH, _EMAIL_BOOK_PH}


# ---------------------------------------------------------------------------
# Helper: Luhn check
# ---------------------------------------------------------------------------

def _luhn_check(number: str) -> bool:
    """Return True if the digit string passes the Luhn algorithm."""
    digits = [int(c) for c in reversed(number) if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Helper: Shannon entropy
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Secrets: high-entropy tokens near secret-like words
_SECRET_KEYWORDS = re.compile(
    r"\b(api[_\-\s]?key|secret|token|password|passwd|auth|bearer|private[_\-]?key)\b",
    re.IGNORECASE,
)
# A token that looks like a secret (alphanum + special, >= 20 chars)
_SECRET_PATTERN = re.compile(r"\b[A-Za-z0-9_\-\.\/\+]{20,}\b")

# Credit cards: 13-19 digit groups (with spaces or dashes)
_CARD_PATTERN = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")

# Email: standard + obfuscated forms
# "name at gmail dot com", "name[at]domain[dot]com", "name(at)domain.com"
_EMAIL_STANDARD = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_EMAIL_OBFUSCATED = re.compile(
    r"\b[A-Za-z0-9._%+\-]+\s*(?:\[at\]|\(at\)|at)\s*[A-Za-z0-9.\-]+\s*(?:\[dot\]|\(dot\)|dot)\s*[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
# Bracketed: "name <at> domain <dot> com"
_EMAIL_BRACKETED = re.compile(
    r"\b[A-Za-z0-9._%+\-]+\s*<at>\s*[A-Za-z0-9.\-]+\s*<dot>\s*[A-Za-z]{2,}\b",
    re.IGNORECASE,
)

# Nepali mobile: 96x, 97x, 98x - optional +977 or 977 prefix
_NP_MOBILE = re.compile(
    r"(?:\+977[\s\-]?|977[\s\-]?)?(?:96|97|98)\d[\s\-]?\d{3}[\s\-]?\d{4}\b"
)

# Kathmandu landline: 01-XXXXXXX (7 digits after area code)
_KTM_LANDLINE = re.compile(r"\b01[\s\-]?\d{7}\b")

# Generic international phone (10-15 digits, optionally with +/dashes/spaces)
# Requires a context word nearby (phone, tel, mobile, contact, call)
_INTL_PHONE = re.compile(r"\b(?:\+?[\d][\d\s\-\(\)]{8,14}\d)\b")
_PHONE_CONTEXT = re.compile(
    r"\b(phone|tel|telephone|mobile|cell|contact|call|number|fax|whatsapp)\b",
    re.IGNORECASE,
)

# Nepal citizenship certificate: digits-digits-digits with "citizenship" nearby
# NOTE: This is an approximation; official format not verified.
_CITIZENSHIP = re.compile(r"\b\d{2,3}[\-/]\d{2,3}[\-/]\d{4,8}\b")
_CITIZENSHIP_CONTEXT = re.compile(
    r"\b(citizenship|nagarikta|nag|citizen|passport|national\s*id|natid|id\s*number|identification)\b",
    re.IGNORECASE,
)

# PAN (Nepal): typically 9 digit number with "PAN" or "permanent account" nearby
_PAN = re.compile(r"\b\d{9}\b")
_PAN_CONTEXT = re.compile(r"\b(pan|permanent\s+account|tax\s+id|vat\s+no|irb)\b", re.IGNORECASE)

# Bank account: 9-20 digits with bank-related keyword
_BANK_ACCT = re.compile(r"\b\d{9,20}\b")
_BANK_CONTEXT = re.compile(r"\b(account|acc|bank|acct|saving|current|esewa|khalti)\b", re.IGNORECASE)

# IPv4
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")


def _context_present(text: str, start: int, pattern: re.Pattern[str], window: int = 80) -> bool:
    """Return True if the context keyword appears within 'window' chars of 'start'."""
    lo = max(0, start - window)
    hi = min(len(text), start + window)
    return bool(pattern.search(text[lo:hi]))


def _is_already_masked(text: str, start: int, end: int) -> bool:
    """Return True if the span is already a placeholder."""
    region = text[start:end]
    for ph in _PLACEHOLDERS:
        if ph in region:
            return True
    return False


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_pii(text: str) -> list[Finding]:
    """
    Return a list of PII Findings found in text.
    Does NOT modify text; call mask_pii() for that.
    """
    findings: list[Finding] = []
    # Collect spans already covered to avoid double-reporting
    covered: list[tuple[int, int]] = []

    def _add(ft: FindingType, start: int, end: int, preview: str) -> None:
        for cs, ce in covered:
            if start < ce and end > cs:
                return  # overlaps with existing finding
        findings.append(Finding(type=ft, span=(start, end), masked_preview=preview))
        covered.append((start, end))

    # 1. Secrets (high-entropy token near keyword)
    for m in _SECRET_PATTERN.finditer(text):
        token = m.group(0)
        if token in _PLACEHOLDERS:
            continue
        ent = _shannon_entropy(token)
        if ent >= 4.0 and _context_present(text, m.start(), _SECRET_KEYWORDS):
            _add(FindingType.PII_SECRET, m.start(), m.end(), _SECRET_PH)

    # 2. Credit/debit cards (Luhn)
    for m in _CARD_PATTERN.finditer(text):
        digits = re.sub(r"[\s\-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_check(digits):
            if not _is_already_masked(text, m.start(), m.end()):
                _add(FindingType.PII_CARD, m.start(), m.end(), _CARD_PH)

    # 3. Email (standard)
    for m in _EMAIL_STANDARD.finditer(text):
        if not _is_already_masked(text, m.start(), m.end()):
            _add(FindingType.PII_EMAIL, m.start(), m.end(), _EMAIL_PH)

    # 4. Email (obfuscated "at" / "dot" forms)
    for m in _EMAIL_OBFUSCATED.finditer(text):
        if not _is_already_masked(text, m.start(), m.end()):
            _add(FindingType.PII_EMAIL, m.start(), m.end(), _EMAIL_PH)

    # 5. Email (bracketed <at> <dot> form)
    for m in _EMAIL_BRACKETED.finditer(text):
        if not _is_already_masked(text, m.start(), m.end()):
            _add(FindingType.PII_EMAIL, m.start(), m.end(), _EMAIL_PH)

    # 6. Nepali mobile numbers
    for m in _NP_MOBILE.finditer(text):
        if not _is_already_masked(text, m.start(), m.end()):
            _add(FindingType.PII_PHONE, m.start(), m.end(), _PHONE_PH)

    # 7. Kathmandu landlines
    for m in _KTM_LANDLINE.finditer(text):
        if not _is_already_masked(text, m.start(), m.end()):
            _add(FindingType.PII_PHONE, m.start(), m.end(), _PHONE_PH)

    # 8. Generic international phone (requires context word)
    for m in _INTL_PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 15:
            if _context_present(text, m.start(), _PHONE_CONTEXT):
                if not _is_already_masked(text, m.start(), m.end()):
                    _add(FindingType.PII_PHONE, m.start(), m.end(), _PHONE_PH)

    # 9. Nepal citizenship certificate (context-gated)
    for m in _CITIZENSHIP.finditer(text):
        if _context_present(text, m.start(), _CITIZENSHIP_CONTEXT):
            if not _is_already_masked(text, m.start(), m.end()):
                _add(FindingType.PII_ID, m.start(), m.end(), _ID_PH)

    # 10. PAN (context-gated)
    for m in _PAN.finditer(text):
        if _context_present(text, m.start(), _PAN_CONTEXT):
            if not _is_already_masked(text, m.start(), m.end()):
                _add(FindingType.PII_ID, m.start(), m.end(), _ID_PH)

    # 11. Bank account (context-gated)
    for m in _BANK_ACCT.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 9 <= len(digits) <= 20:
            if _context_present(text, m.start(), _BANK_CONTEXT):
                if not _is_already_masked(text, m.start(), m.end()):
                    _add(FindingType.PII_ID, m.start(), m.end(), _ID_PH)

    # 12. IPv4
    for m in _IPV4.finditer(text):
        if not _is_already_masked(text, m.start(), m.end()):
            _add(FindingType.PII_IP, m.start(), m.end(), _IP_PH)

    return findings


def mask_pii(text: str, findings: list[Finding] | None = None) -> str:
    """
    Replace PII spans with typed placeholders.
    Accepts pre-computed findings (from detect_pii) or computes them.
    Idempotent: masking already-masked text returns the same text.
    """
    if findings is None:
        findings = detect_pii(text)
    if not findings:
        return text

    # Sort by start position (descending) to replace from right to left
    sorted_findings = sorted(
        [f for f in findings if f.span is not None],
        key=lambda f: f.span[0],  # type: ignore[index]
        reverse=True,
    )

    for finding in sorted_findings:
        if finding.span is None:
            continue
        start, end = finding.span
        placeholder = finding.masked_preview or _placeholder_for(finding.type)
        text = text[:start] + placeholder + text[end:]

    return text


def _placeholder_for(ft: FindingType) -> str:
    mapping = {
        FindingType.PII_EMAIL: _EMAIL_PH,
        FindingType.PII_PHONE: _PHONE_PH,
        FindingType.PII_CARD: _CARD_PH,
        FindingType.PII_ID: _ID_PH,
        FindingType.PII_SECRET: _SECRET_PH,
        FindingType.PII_IP: _IP_PH,
    }
    return mapping.get(ft, "[PII]")


# ---------------------------------------------------------------------------
# Booking email tokenization (U4)
# ---------------------------------------------------------------------------

def tokenize_email_for_booking(text: str) -> tuple[str, str | None]:
    """
    Extract the first email from text, replace it with [EMAIL_1],
    and return (tokenized_text, real_email_or_None).

    The real email is returned out-of-band so it can be passed directly
    to the booking service without going through the LLM.
    """
    m = _EMAIL_STANDARD.search(text)
    if not m:
        return text, None
    real_email = m.group(0)
    tokenized = text[: m.start()] + _EMAIL_BOOK_PH + text[m.end() :]
    return tokenized, real_email
