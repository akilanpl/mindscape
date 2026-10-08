"""Strict JSON decoding for typed core records."""
from dataclasses import fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import get_args, get_origin, get_type_hints, Union


def decode(cls, value):
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, UnionType):
        for candidate in args:
            try:
                return decode(candidate, value)
            except (ValueError, TypeError):
                pass
        raise ValueError("Value does not match union")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError("Expected sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], item) for item in value)
        if len(value) != len(args):
            raise ValueError("Wrong tuple length")
        return tuple(decode(t, v) for t, v in zip(args, value))
    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls(value)
    if is_dataclass(cls):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("Unexpected or missing schema fields")
        hints = get_type_hints(cls)
        return cls(**{key: decode(hints[key], item) for key, item in value.items()})
    if type(value) is not cls:
        raise ValueError(f"Expected {cls}, got {type(value)}")
    return value
