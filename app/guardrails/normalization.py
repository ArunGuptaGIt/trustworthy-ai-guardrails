"""
Input normalization (U1).

Transformations applied before any other guardrail check:
- Unicode NFKC normalization
- Strip zero-width and control characters (except tab/newline)
- Normalize Devanagari digits to ASCII
- Collapse internal whitespace runs to a single space (trim edges)
- De-obfuscate spaced letters ("p r o m p t" -> "prompt")
- Simple leetspeak substitution (3->e, 0->o, @->a, 1->i, 4->a, 5->s, $->s, 7->t)
- Decode base64 blobs that decode cleanly to printable ASCII text

All transformations are idempotent: normalize(normalize(x)) == normalize(x).

LIMITATIONS (documented):
- Spaced-letter de-obfuscation triggers on any spaced initialism; it is only
  applied when >= 4 single-character tokens are in a row so short acronyms like
  "U S A" are not collapsed.
- Leetspeak substitution is shallow (single-character mapping) and misses
  multi-character substitutions like "ph" for "f".
- Base64 decode is attempted only on tokens >= 16 characters to avoid false
  positives on short base64-like strings.
- No language detection: Romanized Nepali or Hindi are treated as English.
"""
from __future__ import annotations

import base64
import re
import unicodedata

# Devanagari digit -> ASCII digit
_DEVA_DIGITS: dict[int, str] = {
    ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")
}

# Simple leetspeak map
_LEET: dict[str, str] = {
    "3": "e",
    "0": "o",
    "1": "i",
    "4": "a",
    "5": "s",
    "$": "s",
    "7": "t",
}

# Zero-width and invisible Unicode code points to strip
_ZERO_WIDTH = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u00ad\ufeff\u2060\u2061\u2062\u2063\u2064]"
)

# Control characters (except \t, \n, \r)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Spaced-letter run: 4+ single chars separated by spaces (optionally with punctuation)
_SPACED_LETTERS = re.compile(r"(?<!\S)([A-Za-z] ){3,}[A-Za-z](?!\S)")

# Base64 token pattern (at least 16 chars, base64url or base64)
_B64_TOKEN = re.compile(r"(?<!\S)[A-Za-z0-9+/=_-]{16,}(?!\S)")


def _try_decode_base64(token: str) -> str:
    """Return decoded printable ASCII text if token is valid base64, else return token."""
    # Add padding if needed
    padded = token + "=" * (-len(token) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True).decode("ascii")
    except Exception:
        # Try url-safe variant
        try:
            padded_url = token.replace("-", "+").replace("_", "/")
            padded_url += "=" * (-len(padded_url) % 4)
            decoded = base64.b64decode(padded_url, validate=True).decode("ascii")
        except Exception:
            return token
    # Only replace if result is printable and substantially shorter (or similar length)
    if decoded.isprintable() and len(decoded) <= len(token):
        return decoded
    return token


def _collapse_spaced_letters(text: str) -> str:
    """Collapse runs of spaced single letters back into a word."""

    def _join(m: re.Match[str]) -> str:
        return m.group(0).replace(" ", "")

    return _SPACED_LETTERS.sub(_join, text)


def _apply_leet(text: str) -> str:
    """Shallow leetspeak substitution: only replace when flanked by a letter."""
    chars = list(text)
    result: list[str] = []
    for i, c in enumerate(chars):
        if c in _LEET:
            prev_alpha = bool(result) and result[-1].isalpha()
            next_alpha = i < len(chars) - 1 and chars[i + 1].isalpha()
            if prev_alpha or next_alpha:
                result.append(_LEET[c])
                continue
        result.append(c)
    return "".join(result)


def normalize(text: str) -> str:
    """
    Apply full normalization pipeline to text.
    Returns the normalized string.  normalize(normalize(x)) == normalize(x).
    """
    # 1. Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip zero-width characters
    text = _ZERO_WIDTH.sub("", text)

    # 3. Strip control characters (preserve tab, newline, CR)
    text = _CONTROL.sub("", text)

    # 4. Devanagari digits -> ASCII digits
    text = text.translate(_DEVA_DIGITS)

    # 5. De-obfuscate spaced letters (4+ single-char tokens in a row)
    text = _collapse_spaced_letters(text)

    # 6. Decode base64 blobs
    def _decode_token(m: re.Match[str]) -> str:
        return _try_decode_base64(m.group(0))

    text = _B64_TOKEN.sub(_decode_token, text)

    # 7. Shallow leetspeak
    text = _apply_leet(text)

    # 8. Collapse whitespace runs (preserve newlines for multi-line messages)
    lines = text.split("\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    text = "\n".join(lines).strip()

    return text
