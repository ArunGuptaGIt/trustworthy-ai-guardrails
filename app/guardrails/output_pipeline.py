"""
Output guardrail pipeline (Phase 3.4).

Stage order for RAG answers:
  1. Schema validation (parse JSON, faithful citations)
  2. Groundedness validation (IDF-weighted claim support)
  3. Sensitive-info detection (PII provenance, canary, prompt-overlap)

Regeneration: capped by config.max_regenerations (default 1).
Fallback: fixed safe message when all retries are exhausted.

For booking extraction: validate BookingExtractStrict, retry once with
a corrective prompt, then ask the user to rephrase.

Monitor mode: all checks run and log, but the original answer is passed through.
Fail-closed: any exception in enforce mode -> FALLBACK.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.guardrails.audit import emit_audit_event
from app.guardrails.config import GuardrailConfig
from app.guardrails.groundedness import check_groundedness
from app.guardrails.output_sensitive import check_output_sensitive
from app.guardrails.schema_validator import build_corrective_prompt, parse_rag_response
from app.guardrails.types import Finding, FindingType, OutputAction, OutputDecision
from app.services.vector_store import VectorSearchResult

logger = logging.getLogger("app.guardrails.output")

_FALLBACK_TEXT = "The uploaded documents do not contain information to answer your question."
_SCHEMA_FALLBACK = "I was unable to generate a properly structured response. Please try again."


class OutputGuardrails:
    """
    Orchestrates all output guardrail checks for the RAG path.
    """

    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config

    async def validate_rag_answer(
        self,
        raw_llm_text: str,
        retrieved_hits: list[VectorSearchResult],
        regenerate_fn: Callable[[str], Awaitable[str]],
        session_context: str = "",
        is_booking_reply: bool = False,
        canary_token: str = "",
        system_prompt: str = "",
        use_professional_threshold: bool = False,
        session_id: str = "",
    ) -> OutputDecision:
        """
        Validate a RAG answer through schema, groundedness, and sensitive-info checks.

        Args:
            raw_llm_text: the raw text returned by the LLM
            retrieved_hits: search results from the vector store
            regenerate_fn: async callable that generates a new response given a prompt
            session_context: concatenated user messages + booking state for provenance
            is_booking_reply: True if this is a booking flow reply
            canary_token: per-request random token to detect system-prompt leakage
            system_prompt: the actual system prompt sent to the LLM (for overlap check)
            use_professional_threshold: use stricter groundedness threshold
            session_id: for audit logging

        Returns:
            OutputDecision with final_text, faithful_citations, action, findings
        """
        if not self.config.enabled:
            return OutputDecision(
                action=OutputAction.PASS,
                final_text=raw_llm_text,
                faithful_citations=[],
                findings=[],
                attempts=1,
            )

        try:
            return await self._run_validation(
                raw_llm_text,
                retrieved_hits,
                regenerate_fn,
                session_context,
                is_booking_reply,
                canary_token,
                system_prompt,
                use_professional_threshold,
                session_id,
            )
        except Exception as exc:
            logger.error("Output guardrail error: %s", exc, exc_info=True)
            if self.config.mode == "enforce":
                return OutputDecision(
                    action=OutputAction.FALLBACK,
                    final_text=self.config.messages.get("output_fallback", _FALLBACK_TEXT),
                    faithful_citations=[],
                    findings=[Finding(type=FindingType.OUTPUT_SCHEMA_INVALID, detail=f"guardrail_error: {exc}")],
                    attempts=1,
                )
            else:
                return OutputDecision(
                    action=OutputAction.PASS,
                    final_text=raw_llm_text,
                    findings=[],
                    attempts=1,
                )

    async def _run_validation(
        self,
        raw_llm_text: str,
        retrieved_hits: list[VectorSearchResult],
        regenerate_fn: Callable[[str], Awaitable[str]],
        session_context: str,
        is_booking_reply: bool,
        canary_token: str,
        system_prompt: str,
        use_professional_threshold: bool,
        session_id: str,
    ) -> OutputDecision:
        all_findings: list[Finding] = []
        attempts = 1
        current_text = raw_llm_text

        # Build lookup for citation faithfulness
        chunk_keys: set[tuple[str, int]] = {(h.document_id, h.chunk_index) for h in retrieved_hits}
        chunk_texts = [h.text for h in retrieved_hits]
        faithful_citations: list[dict[str, Any]] = []

        # ------------------------------------------------------------------
        # STAGE 1: Schema validation
        # ------------------------------------------------------------------
        schema, schema_findings, _err = parse_rag_response(current_text, chunk_keys)
        all_findings.extend(schema_findings)

        if schema is None:
            # Retry once
            if attempts <= self.config.max_regenerations:
                attempts += 1
                corrective = build_corrective_prompt()
                try:
                    current_text = await regenerate_fn(corrective)
                except Exception as exc:
                    logger.warning("Regeneration failed: %s", exc)
                    current_text = ""

                schema, retry_findings, _err = parse_rag_response(current_text, chunk_keys)
                all_findings.extend(retry_findings)

            if schema is None:
                self._emit_audit(session_id, "schema", OutputAction.FALLBACK, all_findings, current_text)
                fallback = self.config.messages.get("schema_fallback", _SCHEMA_FALLBACK)
                return OutputDecision(
                    action=OutputAction.FALLBACK,
                    final_text=fallback,
                    faithful_citations=[],
                    findings=all_findings,
                    attempts=attempts,
                    answerable=False,
                )

        self._emit_audit(session_id, "schema", OutputAction.PASS, schema_findings, current_text)

        # Build faithful citations from schema
        faithful_citations = [{"document_id": c.document_id, "chunk_index": c.chunk_index} for c in schema.citations]
        answer_text = schema.answer

        # ------------------------------------------------------------------
        # STAGE 2: Groundedness
        # ------------------------------------------------------------------
        ground_min = self.config.groundedness_min_claim_support
        ground_redact = self.config.groundedness_redact_threshold
        if use_professional_threshold:
            ground_min = self.config.groundedness_professional_threshold
            ground_redact = self.config.groundedness_professional_threshold

        ground_result = check_groundedness(
            answer_text,
            chunk_texts,
            min_support=ground_min,
            redact_threshold=ground_redact,
        )
        all_findings.extend(ground_result.findings)

        if ground_result.action == OutputAction.REGENERATE:
            # Retry once with a stricter prompt
            if attempts <= self.config.max_regenerations:
                attempts += 1
                strict_prompt = (
                    "The previous response contained claims not supported by the retrieved context. "
                    "Please answer again using ONLY information explicitly stated in the provided documents. "
                    "If the documents do not contain the answer, say so clearly.\n\n" + build_corrective_prompt()
                )
                try:
                    regenerated = await regenerate_fn(strict_prompt)
                    # Re-validate schema
                    retry_schema, retry_sfindings, _ = parse_rag_response(regenerated, chunk_keys)
                    all_findings.extend(retry_sfindings)
                    if retry_schema is not None:
                        retry_ground = check_groundedness(
                            retry_schema.answer,
                            chunk_texts,
                            min_support=ground_min,
                            redact_threshold=ground_redact,
                        )
                        all_findings.extend(retry_ground.findings)
                        if retry_ground.action in (OutputAction.PASS, OutputAction.REDACT):
                            answer_text = retry_ground.final_text
                            faithful_citations = [
                                {"document_id": c.document_id, "chunk_index": c.chunk_index}
                                for c in retry_schema.citations
                            ]
                            ground_result = retry_ground
                except Exception as exc:
                    logger.warning("Groundedness regeneration failed: %s", exc)

            # After retry: if still bad, FALLBACK
            if ground_result.action == OutputAction.REGENERATE:
                self._emit_audit(session_id, "groundedness", OutputAction.FALLBACK, ground_result.findings, answer_text)
                fallback = self.config.messages.get("output_fallback", _FALLBACK_TEXT)
                return OutputDecision(
                    action=OutputAction.FALLBACK,
                    final_text=fallback,
                    faithful_citations=[],
                    findings=all_findings,
                    attempts=attempts,
                    answerable=False,
                )

        answer_text = ground_result.final_text
        ground_action = ground_result.action
        self._emit_audit(session_id, "groundedness", ground_action, ground_result.findings, answer_text)

        # ------------------------------------------------------------------
        # STAGE 3: Sensitive-info detection
        # ------------------------------------------------------------------
        final_text, sens_findings, sens_action = check_output_sensitive(
            answer_text,
            chunk_texts,
            session_context=session_context,
            allow_pii_from_context=self.config.allow_pii_from_context,
            is_booking_reply=is_booking_reply,
            canary_token=canary_token,
            system_prompt=system_prompt,
            ngram_size=self.config.canary_ngram_size,
            overlap_threshold=self.config.canary_overlap_threshold,
        )
        all_findings.extend(sens_findings)
        self._emit_audit(session_id, "output_sensitive", sens_action, sens_findings, final_text)

        if sens_action == OutputAction.REFUSE:
            # Prompt/canary leak or secrets: refuse entirely
            return OutputDecision(
                action=OutputAction.REFUSE,
                final_text=self.config.messages.get("output_fallback", _FALLBACK_TEXT),
                faithful_citations=[],
                findings=all_findings,
                attempts=attempts,
                answerable=False,
            )

        # Determine final action
        if sens_action == OutputAction.REDACT or ground_action == OutputAction.REDACT:
            final_action = OutputAction.REDACT
        elif ground_action == OutputAction.PASS and sens_action == OutputAction.PASS:
            final_action = OutputAction.PASS
        else:
            final_action = OutputAction.PASS

        return OutputDecision(
            action=final_action,
            final_text=final_text,
            faithful_citations=faithful_citations,
            findings=all_findings,
            attempts=attempts,
            answerable=schema.answerable if schema else False,
        )

    def _emit_audit(
        self,
        session_id: str,
        guardrail: str,
        action: OutputAction,
        findings: list[Finding],
        text: str,
    ) -> None:
        emit_audit_event(
            session_id=session_id,
            stage="output",
            guardrail=guardrail,
            action=action.value,
            risk_score=0.0,
            findings=findings,
            message=text,
            policy_version=self.config.policy_version,
        )
