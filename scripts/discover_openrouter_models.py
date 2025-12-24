"""scripts/discover_openrouter_models.py

Fetch OpenRouter models, filter best-effort "free" models, mark "popular" candidates,
and write results to docs/openrouter_free_models.md and models/openrouter_free_models.json.

Usage:
    python scripts/discover_openrouter_models.py

Requires OPENROUTER_API_KEY in environment or will exit with instructions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# When running this script directly from the scripts/ directory, ensure the
# project root is on sys.path so we can import sibling modules like
# `openrouter_client` and `models.model_registry`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openrouter_client import OpenRouterClient
from models.model_registry import save_models


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


def is_popular(model_id: str, provider: str | None) -> bool:
    if not model_id:
        return False
    s = model_id.lower()
    if provider:
        s = s + " " + str(provider).lower()
    return any(k in s for k in KNOWN_POPULAR_KEYWORDS)


def normalize_context(ctx: Any) -> int | None:
    try:
        if ctx is None:
            return None
        if isinstance(ctx, (int, float)):
            return int(ctx)
        s = str(ctx)
        # digits only
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None
    except Exception:
        return None


def generate_markdown(models: List[Dict[str, Any]], out_path: Path, title: str = "OpenRouter: Free Models") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "> 自动生成：列出 OpenRouter 上被识别为“免费”的模型（best-effort）。", ""]
    lines.append("注意：免费模型的标记依赖于各模型元数据字段，可能并不完美，请在生产使用前核对提供者说明。")
    lines.append("")
    lines.append("| Model ID | Provider | Context | Popular | Notes |")
    lines.append("|---|---:|---:|---:|---|")

    for m in models:
        mid = m.get("id") or "<unknown>"
        prov = m.get("provider") or ""
        ctx = m.get("context") or ""
        pop = "✅" if m.get("popular") else ""
        notes = []
        if m.get("is_free"):
            notes.append("free")
        if isinstance(m.get("raw"), dict) and m.get("raw").get("tags"):
            tags = ",".join(m.get("raw").get("tags"))
            notes.append(tags)
        notes_s = "; ".join(notes)
        lines.append(f"| `{mid}` | {prov} | {ctx} | {pop} | {notes_s} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**生成说明**：本文件由 `scripts/discover_openrouter_models.py` 自动生成。请注意隐私与限额。")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-md", default="docs/openrouter_free_models.md")
    parser.add_argument("--out-json", default="models/openrouter_free_models.json")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable and retry.")
        print("PowerShell 示例： $env:OPENROUTER_API_KEY = 'sk-...'  (仅当前会话)")
        return 2

    client = OpenRouterClient()
    print("Fetching models from OpenRouter...")
    models = client.list_models(free_only=True)
    if not models:
        print("未发现免费模型或请求失败。")
        return 1

    # Add derived fields
    processed: List[Dict[str, Any]] = []
    for m in models:
        mid = (m.get("id") or "").strip()
        prov = m.get("provider")
        ctx = normalize_context(m.get("context"))
        popular = is_popular(mid, prov)
        processed.append({
            "id": mid,
            "provider": prov,
            "context": ctx,
            "is_free": m.get("is_free", False),
            "popular": popular,
            "raw": m.get("raw"),
        })

    # Sort: popular first, then context desc
    processed.sort(key=lambda x: ((1 if x.get("popular") else 0), x.get("context") or 0), reverse=True)

    # Save JSON
    save_models(processed, args.out_json)

    # Save Markdown
    generate_markdown(processed, Path(args.out_md))

    print(f"Found {len(processed)} free models (best-effort). Wrote {args.out_md} and {args.out_json}.")
    print("Top 10:")
    for p in processed[:10]:
        print(f" - {p.get('id')} ({p.get('provider')}) context={p.get('context')} popular={p.get('popular')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
