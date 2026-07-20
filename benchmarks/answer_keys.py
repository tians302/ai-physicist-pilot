"""Answer-key loader with verification enforcement (WP5/WP6).

Scoring may use ONLY verified entries (PLAN.md provenance rules). The
loader exposes unverified provisional values separately, for INFO-ONLY
display in notes — never for pass/fail checks.
"""
import json
from pathlib import Path

_KEY_PATH = (Path(__file__).resolve().parent.parent
             / "data" / "reference" / "si_mechanical_answer_key.json")


class UnverifiedKeyError(RuntimeError):
    pass


def load_keys():
    return json.loads(_KEY_PATH.read_text())["entries"]


def verified_entry(name: str) -> dict:
    for e in load_keys():
        if e["name"] == name:
            if not e.get("verified"):
                raise UnverifiedKeyError(
                    f"answer key {name!r} is not verified — scoring refused")
            return e
    raise KeyError(name)


def provisional_info(name: str):
    """INFO-ONLY provisional value (never for scoring)."""
    for e in load_keys():
        if e["name"] == name:
            return e.get("provisional_value")
    return None
