"""
openrouter_client.py

Minimal OpenRouter client utilities for listing models and calling chat completions.

Features:
- OpenRouterClient: init with API key or environment variable
- list_models(free_only=True): returns normalized list of models
- chat_completion(...): thin wrapper around OpenRouter chat/completions endpoint

This file is intentionally small and dependency-light (uses requests).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from config import OPENROUTER


logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class OpenRouterClient:
    """Tiny client for OpenRouter API.

    Usage:
        client = OpenRouterClient()  # reads OPENROUTER_API_KEY from env if not passed
        models = client.list_models(free_only=True)
        client.chat_completion(model="olmo-3.1-32b", messages=[{"role": "user", "content": "hello"}])
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get(OPENROUTER.get("API_KEY_ENV", "OPENROUTER_API_KEY"))
        if not self.api_key:
            logger.warning(
                "OpenRouter API key not provided. Some calls will fail unless you set OPENROUTER_API_KEY in env."
            )
        self.base_url = base_url or OPENROUTER.get("API_BASE", "https://openrouter.ai/api/v1")
        self.timeout = timeout

    # ---------------------- Helpers ----------------------
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ---------------------- Models ----------------------
    def list_models(self, free_only: bool = True) -> List[Dict[str, Any]]:
        """List models available on OpenRouter.

        Attempts to call the API. If the API doesn't return an expected shape,
        returns an empty list and logs useful debugging info.

        free_only: If True, try to filter models that are free (price==0 or marked as free).
        """
        endpoint = f"{self.base_url}/models"
        try:
            resp = requests.get(endpoint, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            # Provide more context for debug: status code and truncated body when available
            status = None
            body = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    status = exc.response.status_code
                    body = exc.response.text[:800]
                except Exception:
                    pass
            logger.error(
                "Failed to fetch models from OpenRouter: %s (status=%s) body=%s",
                exc,
                status,
                body,
            )
            return []

        try:
            data = resp.json()
        except ValueError:
            logger.error("OpenRouter returned non-json response for %s", endpoint)
            return []

        # The API may return {'models': [...]} or {'data': [...]} or a list directly. Normalize.
        if isinstance(data, dict) and "models" in data:
            models_raw = data.get("models")
        elif isinstance(data, dict) and "data" in data:
            models_raw = data.get("data")
        else:
            models_raw = data

        if not isinstance(models_raw, list):
            logger.debug("Unexpected models response shape: %s", type(models_raw))
            return []

        normalized: List[Dict[str, Any]] = []
        for item in models_raw:
            # Best-effort normalization. Fields vary by provider.
            model_id = item.get("id") or item.get("model") or item.get("name")
            provider = item.get("provider") or item.get("owner") or item.get("publisher")
            context = item.get("max_context") or item.get("context_length") or item.get("context")
            pricing = item.get("pricing") or item.get("price") or item.get("pricing_info") or {}

            # Determine whether it's free (best-effort): price fields 0, numeric strings '0', or explicit free tag
            is_free = False
            try:
                if isinstance(pricing, dict):
                    # try to interpret numeric strings as numbers
                    numeric_vals = []
                    for v in pricing.values():
                        if isinstance(v, (int, float)):
                            numeric_vals.append(v)
                        else:
                            try:
                                # convert numeric strings like "0.0000003" or "0" to float
                                nv = float(str(v))
                                numeric_vals.append(nv)
                            except Exception:
                                pass
                    if numeric_vals and all(v == 0 for v in numeric_vals):
                        is_free = True
                elif isinstance(pricing, (int, float)) and pricing == 0:
                    is_free = True
                else:
                    # pricing might be '0' string
                    if str(pricing).strip() == '0':
                        is_free = True
            except Exception:
                pass

            # Some API descriptions include a 'free' boolean or tag
            if not is_free:
                if isinstance(item.get("tags"), list) and "free" in [t.lower() for t in item.get("tags")]:
                    is_free = True
                if str(item.get("free", "")).lower() in ("true", "1"):
                    is_free = True

            normalized.append(
                {
                    "id": model_id,
                    "raw": item,
                    "provider": provider,
                    "context": context,
                    "pricing": pricing,
                    "is_free": is_free,
                }
            )

        if free_only:
            return [m for m in normalized if m.get("is_free")]
        return normalized

    # ---------------------- Chat completions ----------------------
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        response_format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Call the OpenRouter Chat Completions endpoint.

        Returns the JSON response for a non-streaming call. For streaming support,
        caller may call with stream=True and handle the raw response.
        """
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not set (OPENROUTER_API_KEY)")

        endpoint = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format
        payload.update(kwargs)

        try:
            if stream:
                # For streaming we return the requests.Response so callers can iterate .iter_lines()
                resp = requests.post(endpoint, json=payload, headers=self._headers(), stream=True, timeout=self.timeout)
                resp.raise_for_status()
                return {"streaming_response": resp}
            resp = requests.post(endpoint, json=payload, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("OpenRouter chat completion failed: %s", exc)
            raise


# ---------------------- CLI / quick demo ----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = OpenRouterClient()
    print("Listing free models (best-effort):")
    models = client.list_models(free_only=True)
    if not models:
        print("No models discovered or failed to contact OpenRouter.")
    else:
        for m in models:
            print(json.dumps({"id": m.get("id"), "provider": m.get("provider"), "context": m.get("context")}, ensure_ascii=False))

    # Example chat call (requires OPENROUTER_API_KEY env var or pass api_key param):
    # try:
    #     resp = client.chat_completion(model="olmo-3.1-32b", messages=[{"role":"user","content":"Hello"}])
    #     print(resp)
    # except Exception as e:
    #     print("Chat call failed", e)
