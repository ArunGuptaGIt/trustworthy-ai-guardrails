"""
Core types for the guardrail system.

InputAction precedence (highest wins): HARD_BLOCK > SAFE_RESPONSE > MASK > ALLOW_WITH_DISCLAIMER > ALLOW
OutputAction precedence: REFUSE > REGENERATE > REDACT > PASS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class InputAction(StrEnum):
    """Action to take after input guardrail checks. Higher values = more restrictive."""

    ALLOW = "allow"
    ALLOW_WITH_DISCLAIMER = "allow_with_disclaimer"
    MASK = "mask"
    SAFE_RESPONSE = "safe_response"
    HARD_BLOCK = "hard_block"

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, InputAction):
            return NotImplemented
        order = list(InputAction)
        return order.index(self) > order.index(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, InputAction):
            return NotImplemented
        order = list(InputAction)
        return order.index(self) >= order.index(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, InputAction):
            return NotImplemented
        order = list(InputAction)
        return order.index(self) < order.index(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, InputAction):
            return NotImplemented
        order = list(InputAction)
        return order.index(self) <= order.index(other)


class OutputAction(StrEnum):
    """Action taken during output validation."""

    PASS = "pass"
    REDACT = "redact"
    REGENERATE = "regenerate"
    REFUSE = "refuse"
    FALLBACK = "fallback"


class FindingType(StrEnum):
    """Category of a guardrail finding."""

    # PII
    PII_EMAIL = "pii:email"
    PII_PHONE = "pii:phone"
    PII_CARD = "pii:card"
    PII_ID = "pii:id"
    PII_SECRET = "pii:secret"
    PII_IP = "pii:ip"

    # Injection
    INJECTION_DIRECT = "injection:direct_injection"
    INJECTION_JAILBREAK = "injection:jailbreak"
    INJECTION_EXTRACTION = "injection:extraction"
    INJECTION_IMPERSONATION = "injection:impersonation"
    INJECTION_CONTEXT_MANIPULATION = "injection:context_manipulation"
    INJECTION_ENCODING_TRICK = "injection:encoding_trick"
    INJECTION_EXFILTRATION = "injection:exfiltration"
    INJECTION_MULTI_TURN = "injection:multi_turn"

    # Topics
    TOPIC_SELF_HARM = "topic:self_harm"
    TOPIC_VIOLENCE = "topic:violence_weapons"
    TOPIC_ILLEGAL = "topic:illegal_activity"
    TOPIC_CYBER_ABUSE = "topic:cyber_abuse"
    TOPIC_HATE = "topic:hate_harassment"
    TOPIC_EXFILTRATION = "topic:data_exfiltration"
    TOPIC_PROFESSIONAL = "topic:professional_advice"

    # Output
    OUTPUT_SCHEMA_INVALID = "output:schema_invalid"
    OUTPUT_UNGROUNDED = "output:ungrounded"
    OUTPUT_PII_LEAK = "output:pii_leak"
    OUTPUT_SECRET_LEAK = "output:secret_leak"
    OUTPUT_CANARY_LEAK = "output:canary_leak"
    OUTPUT_PROMPT_OVERLAP = "output:prompt_overlap"
    OUTPUT_CITATION_UNFAITHFUL = "output:citation_unfaithful"

    # Length
    LENGTH_EXCEEDED = "length:exceeded"
    LENGTH_EMPTY = "length:empty"


class PIIProvenance(StrEnum):
    """Where PII in an output came from."""

    SESSION_OWN = "session_own"
    CONTEXT_SOURCED = "context_sourced"
    UNKNOWN = "unknown"


@dataclass
class Finding:
    """A single guardrail finding."""

    type: FindingType
    detail: str = ""
    span: tuple[int, int] | None = None
    masked_preview: str = ""
    weight: float = 1.0
    provenance: PIIProvenance | None = None

    def flag_label(self) -> str:
        """Short label used in ChatResponse.guardrails flags list."""
        return self.type.value


@dataclass
class InputDecision:
    """Result of running all input guardrails."""

    action: InputAction
    sanitized_message: str
    findings: list[Finding] = field(default_factory=list)
    risk_score: float = 0.0
    user_message: str = ""
    # Extra context attached for downstream guardrails (e.g. disclaimer text)
    disclaimer: str = ""
    # Session risk score after this turn (written back to Redis by the pipeline)
    updated_session_risk: float = 0.0

    @property
    def flags(self) -> list[str]:
        return [f.flag_label() for f in self.findings]


@dataclass
class OutputDecision:
    """Result of running all output guardrails."""

    action: OutputAction
    final_text: str
    faithful_citations: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    attempts: int = 1
    answerable: bool = True

    @property
    def flags(self) -> list[str]:
        return [f.flag_label() for f in self.findings]


@dataclass
class GuardrailSummary:
    """Compact summary attached to ChatResponse when guardrails are active."""

    # One of: passed, masked, flagged, blocked, safe_response, redacted, regenerated, fallback
    status: str
    flags: list[str] = field(default_factory=list)
    # Detail only populated when GUARDRAILS_DEBUG=true
    detail: dict[str, Any] | None = None
    policy_version: str = ""
