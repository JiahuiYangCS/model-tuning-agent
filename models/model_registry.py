"""models/model_registry.py

Simple registry utilities for storing and loading model metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def save_models(models: List[Dict[str, Any]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


def load_models(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
