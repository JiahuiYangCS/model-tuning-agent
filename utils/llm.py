"""Unified LLM helper for OpenAI (OpenAI SDK) and OpenRouter client.
Provides a small compatibility shim used by agents to call either provider.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import os

from openrouter_client import OpenRouterClient
from utils.openai_client import client as openai_client


KNOWN_POPULAR_KEYWORDS = [
    "olmo",
    "mimo",
    "nemotron",
    "mistral",
    "llama",
    "alpaca",
    "vicuna",
    "mpt",
    "gpt",
    "gemini",
]


def _load_openrouter_api_key() -> Optional[str]:
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key:
                            return key
        except Exception:
            pass
    return os.environ.get("OPENROUTER_API_KEY")


def chat(
    source: str = "openai",
    model: str = "gpt-3.5-turbo",
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.3,
    **kwargs,
) -> str:
    """Return the content (string) of a chat completion from the chosen provider.

    source: 'openai' or 'openrouter'
    model: model id
    messages: list of message dicts like OpenAI format
    """
    messages = messages or []

    if source == "openai":
        # Use existing OpenAI client (utils.openai_client.client)
        if openai_client is None:
            raise RuntimeError("OpenAI client not initialized (run setup_api_key.py or set OPENAI_API_KEY)")
        completion = openai_client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        # Extract content with a few defensive checks
        try:
            return completion.choices[0].message.content
        except Exception:
            # Fallback to dict access
            try:
                return completion["choices"][0]["message"]["content"]
            except Exception:
                return str(completion)

    if source == "openrouter":
        api_key = _load_openrouter_api_key()
        client = OpenRouterClient(api_key=api_key)
        resp = client.chat_completion(model=model, messages=messages, temperature=temperature, **kwargs)
        # Normalize a few possible shapes
        if isinstance(resp, dict):
            # OpenAI-like
            choices = resp.get("choices")
            if choices and isinstance(choices, list):
                c0 = choices[0]
                if isinstance(c0, dict):
                    msg = c0.get("message")
                    if isinstance(msg, dict) and msg.get("content"):
                        return msg.get("content")
                    # Some providers return {"choices": [{"content": "..."}]}
                    if c0.get("content"):
                        return c0.get("content")
            # Some providers return top-level text
            if "text" in resp and isinstance(resp["text"], str):
                return resp["text"]
            if "output" in resp:
                out = resp["output"]
                if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("content"):
                    return out[0].get("content")
            # Last resort
            return json.dumps(resp)

        # If streaming response or unknown, str() it
        return str(resp)

    raise RuntimeError(f"Unknown LLM source: {source}")


def list_openrouter_free_models(top_n: int = 3) -> List[Dict[str, Any]]:
    """Return up to top_n candidate free OpenRouter models (best-effort popular ranking).

    Attempts to read `models/openrouter_free_models.json` first; if missing and env key
    is present, queries OpenRouter directly.
    """
    # Try local cache
    path = Path(__file__).parent.parent / "models" / "openrouter_free_models.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data[:top_n]
        except Exception:
            pass

    # Try live fetch
    api_key = _load_openrouter_api_key()
    if not api_key:
        raise RuntimeError("OpenRouter API key not found: run setup_openrouter_api_key.py or set OPENROUTER_API_KEY env var")
    client = OpenRouterClient(api_key=api_key)
    models = client.list_models(free_only=True)
    # Mark popular by heuristics
    def score(m: Dict[str, Any]) -> int:
        s = 0
        mid = (m.get("id") or "").lower()
        prov = (str(m.get("provider") or "")).lower()
        if any(k in mid for k in KNOWN_POPULAR_KEYWORDS):
            s += 10
        if any(k in prov for k in KNOWN_POPULAR_KEYWORDS):
            s += 5
        # larger context considered slightly better
        ctx = m.get("context")
        if isinstance(ctx, (int, float)):
            s += int(ctx / 1024)
        return s

    models_sorted = sorted(models, key=score, reverse=True)
    return [
        {
            "id": m.get("id"),
            "provider": m.get("provider"),
            "context": m.get("context"),
            "raw": m.get("raw"),
            "is_free": m.get("is_free", False),
        }
        for m in models_sorted[:top_n]
    ]
