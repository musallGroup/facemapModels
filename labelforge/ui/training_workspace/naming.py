from __future__ import annotations

import re
from pathlib import Path


VERSION_SUFFIX = re.compile(r"^(?P<family>.*?)(?:[_\-\s]?v)(?P<version>\d+)$", re.IGNORECASE)


def model_stem(path_or_name: str) -> str:
    return Path(path_or_name.strip()).stem


def ensure_v1(name: str) -> str:
    stem = model_stem(name).rstrip("_- ")
    if not stem:
        return ""
    if VERSION_SUFFIX.match(stem):
        return stem
    return f"{stem}_v1"


def next_refinement_name(parent: str) -> str:
    stem = model_stem(parent).rstrip("_- ")
    if not stem:
        return ""
    match = VERSION_SUFFIX.match(stem)
    if not match:
        return f"{stem}_v1"
    family = match.group("family").rstrip("_- ")
    return f"{family}_v{int(match.group('version')) + 1}"


def specialized_name(name: str) -> str:
    return ensure_v1(name)
