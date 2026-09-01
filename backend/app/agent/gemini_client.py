"""
Gemini 2.5 Flash client wrapper for the recovery agent.

Encapsulates the Google GenAI SDK call with:
  - configurable API key from GEMINI_API_KEY env var
  - timeout handling
  - structured error reporting
  - no direct execution of financial actions

The client is intentionally simple — no multi-provider abstraction,
no LangChain, no agent framework.
"""

from __future__ import annotations

import json
import logging

import google.genai as genai
from google.genai import types as genai_types

from app.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agent.schemas import AgentFailure, AgentInput, AgentProposal
from app.config import settings

logger = logging.getLogger(__name__)


def _get_client() -> genai.Client:
    """Create a Gemini client from the configured API key."""
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add it to backend/.env or set the environment variable."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def call_gemini(agent_input: AgentInput) -> AgentProposal | AgentFailure:
    """
    Send structured context to Gemini 2.5 Flash and parse the response.

    Returns
    -------
    AgentProposal
        If Gemini returns valid JSON matching the schema.
    AgentFailure
        If the call fails, times out, or returns invalid output.
    """
    try:
        client = _get_client()
    except RuntimeError as e:
        return AgentFailure(
            error_type="config_error",
            error_message=str(e),
        )

    user_prompt = build_user_prompt(agent_input)

    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,  # low temperature for deterministic reasoning
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        error_type = "api_error"
        if "timeout" in str(e).lower() or "deadline" in str(e).lower():
            error_type = "timeout"
        elif "rate" in str(e).lower() or "quota" in str(e).lower():
            error_type = "rate_limit"

        logger.warning("Gemini API call failed: %s", e)
        return AgentFailure(
            error_type=error_type,
            error_message=str(e),
        )

    # Extract text from response.
    raw_text = response.text or ""

    return parse_gemini_response(raw_text)


def parse_gemini_response(raw_text: str) -> AgentProposal | AgentFailure:
    """
    Parse and validate raw LLM text into an AgentProposal.

    Handles:
      - clean JSON
      - JSON wrapped in markdown code fences
      - malformed JSON
      - missing / invalid fields
    """
    text = raw_text.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` ... ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Parse JSON.
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return AgentFailure(
            error_type="malformed_json",
            error_message=f"Failed to parse JSON: {e}",
            raw_response=raw_text,
        )

    if not isinstance(data, dict):
        return AgentFailure(
            error_type="malformed_json",
            error_message=f"Expected a JSON object, got {type(data).__name__}",
            raw_response=raw_text,
        )

    # Validate with Pydantic.
    try:
        proposal = AgentProposal(**data)
    except Exception as e:
        return AgentFailure(
            error_type="validation_error",
            error_message=str(e),
            raw_response=raw_text,
        )

    return proposal
