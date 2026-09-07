"""Small, portable settings file kept beside the application."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from domain import WEIGHTS, validate_weights

PATH = Path(__file__).resolve().parent / "setting.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"


def load_settings(path=PATH):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    profiles = value.get("ollama_profiles")
    if not isinstance(profiles, list) or not profiles:
        profiles = [{"name": "このPC", "endpoint": DEFAULT_ENDPOINT}]
    valid = [p for p in profiles if isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("endpoint"), str)]
    if not valid:
        valid = [{"name": "このPC", "endpoint": DEFAULT_ENDPOINT}]
    names, normalized = set(), []
    for profile in valid:
        name = profile["name"].strip()[:80]
        if name and name not in names:
            names.add(name)
            normalized.append({"name": name, "endpoint": profile["endpoint"].strip()})
    weights = value.get("weights", WEIGHTS)
    try:
        weights = validate_weights(weights)
    except ValueError:
        weights = dict(WEIGHTS)
    active = value.get("active_profile")
    if active not in names:
        active = normalized[0]["name"]
    return {**value, "schema_version": 1, "ollama_profiles": normalized, "active_profile": active, "weights": weights}


def save_settings(changes, path=PATH):
    current = load_settings(path)
    current.update(changes)
    current["weights"] = validate_weights(current.get("weights", WEIGHTS))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".setting-", suffix=".json", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            json.dump(current, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return current

