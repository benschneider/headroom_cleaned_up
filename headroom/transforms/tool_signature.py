"""Content-shape signatures used by compression heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class ToolSignature:
    """Small, deterministic summary of content shape."""

    structure_hash: str
    field_count: int = 0
    has_nested_objects: bool = False
    has_arrays: bool = False
    max_depth: int = 0
    string_field_count: int = 0
    has_error_like_field: bool = False
    has_message_like_field: bool = False

    @classmethod
    def from_items(cls, items: Any) -> "ToolSignature":
        shape = _shape(items)
        encoded = json.dumps(shape, sort_keys=True, separators=(",", ":"))
        return cls(
            structure_hash=hashlib.sha256(encoded.encode()).hexdigest()[:24],
            field_count=_field_count(items),
            has_nested_objects=_has_nested_objects(items),
            has_arrays=_has_arrays(items),
            max_depth=_max_depth(items),
        )


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _shape(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


def _field_count(value: Any) -> int:
    if isinstance(value, dict):
        return len(value) + sum(_field_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(_field_count(v) for v in value)
    return 0


def _has_nested_objects(value: Any) -> bool:
    if isinstance(value, dict):
        return any(isinstance(v, dict) or _has_nested_objects(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_nested_objects(v) for v in value)
    return False


def _has_arrays(value: Any) -> bool:
    if isinstance(value, list):
        return True
    if isinstance(value, dict):
        return any(_has_arrays(v) for v in value.values())
    return False


def _max_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max((_max_depth(v) for v in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_max_depth(v) for v in value), default=0)
    return 0
