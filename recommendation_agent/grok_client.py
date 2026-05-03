"""
Thin wrapper around the Grok API (OpenAI-compatible endpoint).
All LLM calls in the chain go through this module.

Supported providers (auto-detected from environment variables):
  - Grok  (xAI)  : set GROK_API_KEY   → uses https://api.x.ai/v1
  - Groq         : set GROQ_API_KEY   → uses https://api.groq.com/openai/v1
"""

import os
import json
from openai import OpenAI

_client: OpenAI | None = None
_model_default: str = "grok-3-mini"


def get_client() -> OpenAI:
    global _client, _model_default
    if _client is not None:
        return _client

    grok_key  = os.environ.get("GROK_API_KEY", "").strip()
    groq_key  = os.environ.get("GROQ_API_KEY", "").strip()

    if groq_key:
        _client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        _model_default = "llama-3.3-70b-versatile"
        print("  [LLM] Using Groq API (llama-3.3-70b-versatile)")
    elif grok_key:
        _client = OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
        _model_default = "grok-3-mini"
        print("  [LLM] Using Grok (xAI) API")
    else:
        raise EnvironmentError(
            "No LLM API key found.\n"
            "Set one of the following:\n"
            "  export GROK_API_KEY=your_key   (from https://console.x.ai)\n"
            "  export GROQ_API_KEY=your_key   (free tier at https://console.groq.com)"
        )

    return _client


def chat(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    expect_json: bool = False,
) -> str:
    """
    Single LLM call. Returns the assistant's text content.

    Parameters
    ----------
    system_prompt : instructions / persona for this step
    user_prompt   : the actual input for this call (contains prior-step data)
    model         : override model name (defaults to provider default)
    temperature   : lower = more deterministic / structured
    expect_json   : if True, request JSON output format
    """
    client = get_client()
    resolved_model = model or _model_default

    kwargs: dict = {
        "model": resolved_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def chat_json(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict:
    """
    Like chat(), but parses the response as JSON and returns a dict.
    Raises ValueError with the raw text if JSON parsing fails.
    """
    raw = chat(system_prompt, user_prompt, model=model,
               temperature=temperature, expect_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON response:\n{raw}"
        ) from exc
