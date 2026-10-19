"""Tests for output guardrails: schema validation, groundedness, output PII."""

from __future__ import annotations

from app.guardrails.groundedness import (
    check_groundedness,
    split_sentences,
)
from app.guardrails.output_sensitive import (
    check_canary_leak,
    check_output_sensitive,
    classify_pii_provenance,
)
from app.guardrails.schema_validator import parse_rag_response
from app.guardrails.types import (
    FindingType,
    OutputAction,
    PIIProvenance,
)

# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_valid_json(self) -> None:
        raw = (
            '{"answer": "The policy states X.", "answerable": true, '
            '"citations": [{"document_id": "doc1", "chunk_index": 0}], '
            '"confidence": 0.9}'
        )
        keys = {("doc1", 0)}
        schema, findings, err = parse_rag_response(raw, keys)
        assert schema is not None
        assert schema.answerable is True
        assert len(schema.citations) == 1

    def test_code_fenced_json(self) -> None:
        raw = (
            '```json\n{"answer": "X.", "answerable": true, '
            '"citations": [{"document_id": "d", "chunk_index": 0}], '
            '"confidence": 0.8}\n```'
        )
        keys = {("d", 0)}
        schema, _, _ = parse_rag_response(raw, keys)
        assert schema is not None

    def test_invalid_json(self) -> None:
        raw = "This is not JSON at all"
        schema, findings, _ = parse_rag_response(raw, set())
        assert schema is None
        assert any(f.type == FindingType.OUTPUT_SCHEMA_INVALID for f in findings)

    def test_string_false_rejected(self) -> None:
        raw = '{"answer": "X.", "answerable": "false", "citations": [], "confidence": 0.5}'
        schema, findings, _ = parse_rag_response(raw, set())
        # strict mode: "false" (string) should fail validation
        assert schema is None
        assert any(f.type == FindingType.OUTPUT_SCHEMA_INVALID for f in findings)

    def test_unfaithful_citation_dropped(self) -> None:
        raw = (
            '{"answer": "X.", "answerable": true, '
            '"citations": [{"document_id": "doc1", "chunk_index": 0}, '
            '{"document_id": "doc999", "chunk_index": 5}], '
            '"confidence": 0.8}'
        )
        keys = {("doc1", 0)}
        schema, findings, _ = parse_rag_response(raw, keys)
        # doc999 was not retrieved, should be dropped
        assert schema is not None
        assert len(schema.citations) == 1
        assert schema.citations[0].document_id == "doc1"
        assert any(f.type == FindingType.OUTPUT_CITATION_UNFAITHFUL for f in findings)


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


class TestGroundedness:
    def test_supported_answer_passes(self) -> None:
        context = ["The interview fee is Rs. 1,500 per candidate."]
        answer = "The interview fee is Rs. 1,500 per candidate."
        result = check_groundedness(answer, context)
        assert result.action == OutputAction.PASS

    def test_unsupported_number_fails(self) -> None:
        context = ["The fee is Rs. 1,500."]
        answer = "The fee is Rs. 5,000."
        result = check_groundedness(answer, context)
        assert result.action in (OutputAction.REDACT, OutputAction.REGENERATE)

    def test_negation_mismatch(self) -> None:
        from app.guardrails.groundedness import (
            LexicalVerifier,
            _compute_idf,
            split_sentences,
        )

        context = "The office does provide lunch."
        answer = "The office does not provide lunch."
        ctx_sents = split_sentences(context)
        idf = _compute_idf(ctx_sents)
        v = LexicalVerifier()
        score = v.verify(answer, ctx_sents, idf)
        # Negation polarity mismatch should halve the score
        assert score < 1.0

    def test_amount_canonicalization(self) -> None:
        context = ["The fee is Rs. 1,500."]
        answer = "The fee is NPR 1500."
        result = check_groundedness(answer, context)
        assert result.action == OutputAction.PASS

    def test_sentence_splitter_abbreviations(self) -> None:
        text = "The fee is Rs. 1,500. Please pay by Friday."
        sents = split_sentences(text)
        assert len(sents) == 2

    def test_meta_sentence_skipped(self) -> None:
        context = ["The policy covers X."]
        answer = "Based on the provided documents, the policy covers X."
        result = check_groundedness(answer, context)
        assert result.action == OutputAction.PASS


# ---------------------------------------------------------------------------
# Output Sensitive Info
# ---------------------------------------------------------------------------


class TestOutputSensitive:
    def test_canary_leak_detected(self) -> None:
        leaked, detail = check_canary_leak(
            "Sure! The canary is CANARY_TOKEN_123",
            canary_token="CANARY_TOKEN_123",
        )
        assert leaked is True

    def test_prompt_ngram_overlap(self) -> None:
        prompt = "You are a helpful assistant that answers from context only"
        answer = "You are a helpful assistant that answers from context only and more"
        leaked, _ = check_canary_leak(
            answer,
            canary_token="",
            system_prompt=prompt,
            ngram_size=6,
            overlap_threshold=0.3,
        )
        assert leaked is True

    def test_unknown_provenance_redacted(self) -> None:
        answer = "Contact john@secret.com for help"
        text, findings, action = check_output_sensitive(
            answer,
            retrieved_chunks=[],
            session_context="",
            allow_pii_from_context=False,
        )
        assert action == OutputAction.REDACT
        assert "[EMAIL]" in text

    def test_context_sourced_pii_redacted(self) -> None:
        answer = "Contact ram@company.com for details"
        chunks = ["For help, email ram@company.com."]
        text, findings, action = check_output_sensitive(
            answer,
            retrieved_chunks=chunks,
            session_context="",
            allow_pii_from_context=False,
        )
        assert action == OutputAction.REDACT

    def test_context_sourced_pii_allowed(self) -> None:
        answer = "Contact ram@company.com for details"
        chunks = ["For help, email ram@company.com."]
        text, findings, action = check_output_sensitive(
            answer,
            retrieved_chunks=chunks,
            session_context="",
            allow_pii_from_context=True,
        )
        assert action == OutputAction.PASS

    def test_pii_provenance_session_own(self) -> None:
        prov = classify_pii_provenance(
            "alice@example.com",
            session_context="user said alice@example.com",
            retrieved_chunks=[],
        )
        assert prov == PIIProvenance.SESSION_OWN

    def test_pii_provenance_unknown(self) -> None:
        prov = classify_pii_provenance(
            "unknown@evil.com",
            session_context="",
            retrieved_chunks=[],
        )
        assert prov == PIIProvenance.UNKNOWN
