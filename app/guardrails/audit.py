"""
Structured audit logging for the guardrail system.

Every guardrail event is emitted as a structured JSON log through the existing
logger at INFO level.  Raw PII, raw user text, and secret values are NEVER
logged.  Only the sha256 prefix (first 16 hex chars) of the sanitized message
is included so events from the same turn can be correlated without revealing
content.

Audit event schema:
{
    "audit": true,
    "request_id": str,       # from request_id_ctx
    "session_hash": str,     # sha256[:12] of session_id
    "stage": str,            # "input" | "output" | "chunk_scan"
    "guardrail": str,        # "normalization" | "pii" | "injection" | "topics" | ...
    "action": str,           # InputAction or OutputAction value
    "risk_score": float,
    "finding_types": list[str],
    "message_prefix": str,   # sha256[:16] of the (sanitized) message
    "policy_version": str,
}
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.core.logging import request_id_ctx
from app.guardrails.types import Finding

logger = logging.getLogger("app.guardrails.audit")


def _sha256_prefix(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def emit_audit_event(
    *,
    session_id: str,
    stage: str,
    guardrail: str,
    action: str,
    risk_score: float,
    findings: list[Finding],
    message: str,
    policy_version: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a structured audit log event. Never logs raw text or PII values."""
    event: dict[str, Any] = {
        "audit": True,
        "request_id": request_id_ctx.get(),
        "session_hash": _sha256_prefix(session_id, 12),
        "stage": stage,
        "guardrail": guardrail,
        "action": action,
        "risk_score": round(risk_score, 4),
        "finding_types": [f.type.value for f in findings],
        "message_prefix": _sha256_prefix(message),
        "policy_version": policy_version,
    }
    if extra:
        event.update(extra)
    logger.info("AUDIT %s", event)
