"""
Policy configuration for the guardrail system.

Loaded from app/guardrails/policy.toml via tomllib (stdlib in Python 3.11+).
Key thresholds can be overridden by environment variables.

Fail-closed: any exception during guardrail execution in enforce mode blocks
the request with a generic safe message. In monitor mode, exceptions are logged
and the request proceeds unchanged.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_POLICY_PATH = Path(__file__).parent / "policy.toml"


@dataclass
class GuardrailConfig:
    # Master switches
    enabled: bool = True
    mode: str = "enforce"  # "enforce" or "monitor"
    debug: bool = False
    allow_pii_from_context: bool = False

    # Length
    max_message_chars: int = 2000

    # Injection thresholds
    injection_flag_threshold: float = 0.35
    injection_block_threshold: float = 0.65
    injection_mention_discount: float = 0.4
    session_risk_decay: float = 0.7
    session_risk_ttl_seconds: int = 3600

    # Groundedness thresholds
    groundedness_min_claim_support: float = 0.5
    groundedness_redact_threshold: float = 0.7  # below this -> REGENERATE
    groundedness_professional_threshold: float = 0.85

    # Output PII
    canary_ngram_size: int = 8
    canary_overlap_threshold: float = 0.3

    # Policy metadata
    policy_version: str = "1.0.0"
    policy_path: str = str(_POLICY_PATH)

    # Topic policy: category -> action string
    topic_policy: dict[str, str] = field(default_factory=dict)
    # User-facing messages
    messages: dict[str, str] = field(default_factory=dict)
    # Disclaimers
    disclaimers: dict[str, str] = field(default_factory=dict)

    # Max regeneration attempts for output guardrails
    max_regenerations: int = 1

    @classmethod
    def from_toml(cls, path: Path | None = None) -> GuardrailConfig:
        """Load config from policy.toml, using defaults for missing keys."""
        p = path or _POLICY_PATH
        if not p.exists():
            return cls()
        with p.open("rb") as fh:
            raw = tomllib.load(fh)

        cfg = cls()
        # thresholds section
        th = raw.get("thresholds", {})
        if "injection_flag" in th:
            cfg.injection_flag_threshold = float(th["injection_flag"])
        if "injection_block" in th:
            cfg.injection_block_threshold = float(th["injection_block"])
        if "injection_mention_discount" in th:
            cfg.injection_mention_discount = float(th["injection_mention_discount"])
        if "session_risk_decay" in th:
            cfg.session_risk_decay = float(th["session_risk_decay"])
        if "groundedness_min" in th:
            cfg.groundedness_min_claim_support = float(th["groundedness_min"])
        if "groundedness_redact" in th:
            cfg.groundedness_redact_threshold = float(th["groundedness_redact"])
        if "groundedness_professional" in th:
            cfg.groundedness_professional_threshold = float(th["groundedness_professional"])
        if "max_message_chars" in th:
            cfg.max_message_chars = int(th["max_message_chars"])

        # topic policy
        tp = raw.get("topic_policy", {})
        cfg.topic_policy = {k: str(v) for k, v in tp.items()}

        # messages
        msg = raw.get("messages", {})
        cfg.messages = {k: str(v) for k, v in msg.items()}

        # disclaimers
        disc = raw.get("disclaimers", {})
        cfg.disclaimers = {k: str(v) for k, v in disc.items()}

        # metadata
        meta = raw.get("metadata", {})
        if "policy_version" in meta:
            cfg.policy_version = str(meta["policy_version"])

        return cfg

    @classmethod
    def from_settings(cls) -> GuardrailConfig:
        """Load from policy.toml then apply env-var overrides (via app.core.config.settings)."""
        cfg = cls.from_toml()
        # Import here to avoid circular imports at module level
        try:
            from app.core.config import settings

            cfg.enabled = settings.GUARDRAILS_ENABLED
            cfg.mode = settings.GUARDRAILS_MODE
            cfg.debug = settings.GUARDRAILS_DEBUG
            cfg.allow_pii_from_context = settings.ALLOW_PII_FROM_CONTEXT
            cfg.max_message_chars = settings.MAX_MESSAGE_CHARS
            if settings.INJECTION_BLOCK_THRESHOLD > 0:
                cfg.injection_block_threshold = settings.INJECTION_BLOCK_THRESHOLD
            if settings.GROUNDEDNESS_MIN_CLAIM_SUPPORT > 0:
                cfg.groundedness_min_claim_support = settings.GROUNDEDNESS_MIN_CLAIM_SUPPORT
        except Exception:
            pass  # fall back to toml values
        return cfg
