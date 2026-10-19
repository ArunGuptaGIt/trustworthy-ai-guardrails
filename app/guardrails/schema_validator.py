"""
Schema validation for the LLM's RAG JSON output (Phase 3.1).

Parsing strategy:
1. Strip markdown code fences (```json ... ```)
2. Find the first balanced JSON object using a bracket scanner
   (not a greedy regex, to handle nested objects)
3. json.loads()
4. Validate with RAGResponseSchema (strict Pydantic)
5. Semantic validation: every citation must reference a chunk retrieved
   this turn; faithful citations only go into the response

On failure: build a corrective retry prompt (generic wording, does not
echo the invalid output), retry once, then FALLBACK to the safe message.
"""
from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from app.guardrails.schemas import Citation, RAGResponseSchema
from app.guardrails.types import Finding, FindingType

logger = logging.getLogger("app.guardrails.schema_validator")


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` code fences."""
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())
    return text.strip()


def _find_first_json_object(text: str) -> str | None:
    """
    Scan for the first balanced JSON object {…} using a bracket counter.
    Returns the JSON string or None.
    """
    depth = 0
    start = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def parse_rag_response(
    raw_text: str,
    retrieved_chunk_keys: set[tuple[str, int]],
) -> tuple[RAGResponseSchema | None, list[Finding], str]:
    """
    Parse and validate the LLM's raw text as a RAG response.

    Args:
        raw_text: the raw string returned by the LLM
        retrieved_chunk_keys: set of (document_id, chunk_index) for chunks
                              retrieved this turn. Used to validate citations.

    Returns:
        (schema_or_None, findings, error_message)
    """
    findings: list[Finding] = []
    text = _strip_fences(raw_text)
    json_str = _find_first_json_object(text)

    if json_str is None:
        findings.append(Finding(type=FindingType.OUTPUT_SCHEMA_INVALID, detail="no JSON object found"))
        return None, findings, "The response did not contain a JSON object."

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        findings.append(Finding(type=FindingType.OUTPUT_SCHEMA_INVALID, detail=f"json decode: {exc}"))
        return None, findings, f"JSON decode error: {exc}"

    try:
        schema = RAGResponseSchema.model_validate(data)
    except ValidationError as exc:
        detail = "; ".join(str(e["msg"]) for e in exc.errors())
        findings.append(Finding(type=FindingType.OUTPUT_SCHEMA_INVALID, detail=detail))
        return None, findings, f"Schema validation failed: {detail}"

    # Validate citations are faithful (reference only retrieved chunks)
    faithful: list[Citation] = []
    unfaithful: list[Citation] = []
    for cite in schema.citations:
        key = (cite.document_id, cite.chunk_index)
        if key in retrieved_chunk_keys:
            faithful.append(cite)
        else:
            unfaithful.append(cite)
            findings.append(
                Finding(
                    type=FindingType.OUTPUT_CITATION_UNFAITHFUL,
                    detail=f"citation not retrieved: doc={cite.document_id} chunk={cite.chunk_index}",
                )
            )

    # Replace citations with faithful-only list
    schema = schema.model_copy(update={"citations": faithful})

    # Re-check answerable constraint after dropping citations
    if schema.answerable and not faithful:
        findings.append(
            Finding(
                type=FindingType.OUTPUT_SCHEMA_INVALID,
                detail="answerable=true but no faithful citations remain",
            )
        )
        return None, findings, "All citations were unfaithful."

    return schema, findings, ""


def build_corrective_prompt(error_context: str = "") -> str:
    """
    Build a generic corrective retry prompt. Does NOT echo the invalid output
    or mention specific rule details.
    """
    return (
        "Please try again. Return ONLY a valid JSON object in this exact format:\n"
        '{"answer": "your answer text", "answerable": true, '
        '"citations": [{"document_id": "...", "chunk_index": 0}], "confidence": 0.9}\n'
        "Rules:\n"
        "- answerable must be a JSON boolean (true or false), not a string\n"
        "- citations must only reference document IDs and chunk indices from the retrieved context\n"
        "- If the documents do not contain the answer, set answerable to false and explain\n"
        "- Do not include any text outside the JSON object"
    )
