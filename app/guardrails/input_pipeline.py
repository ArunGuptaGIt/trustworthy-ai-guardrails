"""
Input guardrail pipeline (Phase 2.4).

Check order (all checks run, most severe action wins):
1. Length check
2. Normalization
3. PII detection + masking
4. Injection scoring
5. Topic classification

Actions:
  HARD_BLOCK     -> return fixed blocked message, no LLM call
  SAFE_RESPONSE  -> return fixed safe message (self-harm), no LLM/RAG call
  MASK           -> continue with sanitized text, user gets a privacy notice
  ALLOW_WITH_DISCLAIMER -> continue, append disclaimer to output
  ALLOW          -> continue unchanged

Monitor mode: all checks run and log, but action is always ALLOW.
Fail-closed: any exception in enforce mode -> HARD_BLOCK with generic message.
"""
from __future__ import annotations

import logging
from typing import Any

from app.guardrails.audit import emit_audit_event
from app.guardrails.config import GuardrailConfig
from app.guardrails.injection import get_session_risk, score_injection, update_session_risk
from app.guardrails.normalization import normalize
from app.guardrails.pii import detect_pii, mask_pii, tokenize_email_for_booking
from app.guardrails.topics import classify_topic
from app.guardrails.types import Finding, FindingType, InputAction, InputDecision

logger = logging.getLogger("app.guardrails.input")


class InputGuardrails:
    """
    Orchestrates all input guardrail checks.

    Usage:
        decision = await guardrails.check(message, session_id, booking_active=False)
        if decision.action == InputAction.HARD_BLOCK:
            return ChatResponse(answer=decision.user_message, ...)
    """

    def __init__(self, config: GuardrailConfig, redis: Any | None = None) -> None:
        self.config = config
        self.redis = redis  # optional: needed for session-risk persistence

    async def check(
        self,
        message: str,
        session_id: str = "",
        booking_active: bool = False,
        intent: str | None = None,
    ) -> InputDecision:
        """
        Run all input guardrails on 'message'.

        Returns InputDecision with:
        - action: the most severe action triggered
        - sanitized_message: text with PII masked (use this downstream)
        - findings: all individual findings
        - risk_score: injection risk score after session accumulation
        - user_message: human-friendly message to show when blocking/masking
        - updated_session_risk: new session risk score (caller should persist)
        """
        if not self.config.enabled:
            return InputDecision(
                action=InputAction.ALLOW,
                sanitized_message=message,
                user_message="",
            )

        try:
            return await self._run_checks(message, session_id, booking_active, intent)
        except Exception as exc:
            logger.error("Input guardrail error: %s", exc, exc_info=True)
            if self.config.mode == "enforce":
                return InputDecision(
                    action=InputAction.HARD_BLOCK,
                    sanitized_message="",
                    user_message=self.config.messages.get("hard_block", "I cannot help with that request."),
                )
            else:
                logger.warning("Monitor mode: guardrail error, passing through.")
                return InputDecision(
                    action=InputAction.ALLOW,
                    sanitized_message=message,
                    user_message="",
                )

    async def _run_checks(
        self,
        message: str,
        session_id: str,
        booking_active: bool,
        intent: str | None,
    ) -> InputDecision:
        all_findings: list[Finding] = []
        dict.fromkeys(InputAction, 0)
        worst_action = InputAction.ALLOW

        # ------------------------------------------------------------------
        # 1. Length check
        # ------------------------------------------------------------------
        if len(message) > self.config.max_message_chars:
            f = Finding(type=FindingType.LENGTH_EXCEEDED, detail=f"len={len(message)}")
            all_findings.append(f)
            self._emit_audit(session_id, "length", InputAction.HARD_BLOCK, 0.0, [f], message)
            msg = self.config.messages.get("length_exceeded", "Your message is too long.")
            return self._decision(
                action=InputAction.HARD_BLOCK,
                sanitized=message[: self.config.max_message_chars],
                findings=[f],
                risk=0.0,
                user_msg=msg,
                disclaimer="",
                session_risk=0.0,
            )

        if not message.strip():
            f = Finding(type=FindingType.LENGTH_EMPTY)
            all_findings.append(f)
            return self._decision(InputAction.ALLOW, message, [f], 0.0, "", "", 0.0)

        # ------------------------------------------------------------------
        # 2. Normalization
        # ------------------------------------------------------------------
        normalized = normalize(message)

        # ------------------------------------------------------------------
        # 3. PII detection + masking
        # ------------------------------------------------------------------
        pii_findings = detect_pii(normalized)
        sanitized = normalized
        has_secret = any(f.type == FindingType.PII_SECRET for f in pii_findings)
        has_card = any(f.type == FindingType.PII_CARD for f in pii_findings)
        if pii_findings:
            # Booking email tokenization: replace email with placeholder BEFORE masking
            if booking_active:
                sanitized, _real_email = tokenize_email_for_booking(normalized)
                # Re-detect after tokenization to avoid double-masking
                pii_findings = detect_pii(sanitized)
            sanitized = mask_pii(sanitized, pii_findings)
            all_findings.extend(pii_findings)
            worst_action = InputAction.MASK

        self._emit_audit(session_id, "pii", worst_action, 0.0, pii_findings, sanitized)

        # ------------------------------------------------------------------
        # 4. Injection scoring
        # ------------------------------------------------------------------
        prev_session_risk = await get_session_risk(self.redis, session_id) if self.redis else 0.0
        inj_score, inj_findings = score_injection(
            sanitized,
            session_risk=prev_session_risk,
            mention_discount=self.config.injection_mention_discount,
        )
        all_findings.extend(inj_findings)

        inj_action = InputAction.ALLOW
        if inj_score >= self.config.injection_block_threshold:
            inj_action = InputAction.HARD_BLOCK
        elif inj_score >= self.config.injection_flag_threshold:
            inj_action = InputAction.MASK  # flagged: add reinforcement prompt, log

        if inj_action > worst_action:
            worst_action = inj_action

        # Update session risk
        updated_risk = await update_session_risk(
            self.redis,
            session_id,
            inj_score,
            decay=self.config.session_risk_decay,
            ttl_seconds=self.config.session_risk_ttl_seconds,
        ) if self.redis else inj_score

        self._emit_audit(session_id, "injection", inj_action, inj_score, inj_findings, sanitized)

        # ------------------------------------------------------------------
        # 5. Topic classification
        # ------------------------------------------------------------------
        topic_action, topic_findings = classify_topic(sanitized, self.config.topic_policy)
        all_findings.extend(topic_findings)
        if topic_action > worst_action:
            worst_action = topic_action

        self._emit_audit(session_id, "topics", topic_action, 0.0, topic_findings, sanitized)

        # ------------------------------------------------------------------
        # Build user-facing message and disclaimer
        # ------------------------------------------------------------------
        user_msg = ""
        disclaimer = ""

        if worst_action == InputAction.HARD_BLOCK:
            # Check specific topic to use a better message
            self_harm = any(f.type == FindingType.TOPIC_SELF_HARM for f in all_findings)
            if self_harm:
                user_msg = self.config.messages.get(
                    "safe_response_self_harm",
                    "I'm concerned. Please reach out to a trusted person or local support service.",
                )
                worst_action = InputAction.SAFE_RESPONSE
            else:
                inj_blocked = any(
                    f.type.value.startswith("injection:")
                    for f in all_findings
                    if inj_score >= self.config.injection_block_threshold
                )
                if inj_blocked:
                    user_msg = self.config.messages.get(
                        "injection_blocked", "Your message was blocked."
                    )
                else:
                    user_msg = self.config.messages.get(
                        "hard_block", "I cannot help with that request."
                    )

        elif worst_action == InputAction.SAFE_RESPONSE:
            # Self-harm path
            user_msg = self.config.messages.get(
                "safe_response_self_harm",
                "I'm concerned. Please reach out to a trusted person or local support service.",
            )

        elif worst_action == InputAction.MASK:
            flag_th = self.config.injection_flag_threshold
            block_th = self.config.injection_block_threshold
            if inj_score >= flag_th and inj_score < block_th:
                # Flagged: reinforcement prompt added in rag.py
                pass
            notices: list[str] = []
            if pii_findings:
                if has_secret or has_card:
                    notices.append(self.config.messages.get(
                        "secret_notice",
                        "A possible secret was detected and removed.",
                    ))
                else:
                    notices.append(self.config.messages.get(
                        "pii_masked",
                        "I removed some personal information for your privacy.",
                    ))
            user_msg = " ".join(notices)

        elif worst_action == InputAction.ALLOW_WITH_DISCLAIMER:
            disclaimer = self.config.disclaimers.get("professional_advice", "")

        # In monitor mode, override action to ALLOW but keep findings
        if self.config.mode == "monitor" and worst_action != InputAction.ALLOW:
            logger.info("Monitor mode: would have applied %s, passing ALLOW", worst_action.value)
            worst_action = InputAction.ALLOW
            user_msg = ""

        return self._decision(
            action=worst_action,
            sanitized=sanitized,
            findings=all_findings,
            risk=inj_score,
            user_msg=user_msg,
            disclaimer=disclaimer,
            session_risk=updated_risk,
        )

    def _decision(
        self,
        action: InputAction,
        sanitized: str,
        findings: list[Finding],
        risk: float,
        user_msg: str,
        disclaimer: str,
        session_risk: float,
    ) -> InputDecision:
        return InputDecision(
            action=action,
            sanitized_message=sanitized,
            findings=findings,
            risk_score=risk,
            user_message=user_msg,
            disclaimer=disclaimer,
            updated_session_risk=session_risk,
        )

    def _emit_audit(
        self,
        session_id: str,
        guardrail: str,
        action: InputAction,
        risk: float,
        findings: list[Finding],
        message: str,
    ) -> None:
        emit_audit_event(
            session_id=session_id,
            stage="input",
            guardrail=guardrail,
            action=action.value,
            risk_score=risk,
            findings=findings,
            message=message,
            policy_version=self.config.policy_version,
        )
