"""Deterministic fallback for answers whose JSON envelope is malformed.

Models often write unescaped double quotes or raw line breaks inside the
free-text "Analysis" field, which makes the whole object invalid JSON although
the final "Conclusion" is well formed. Taking the LAST "Conclusion": "pos|neg"
pair recovers the decision without model-based repair (amendment A2 of
docs/factorial_openweight_plan.md). A quoted conclusion earlier in the analysis
cannot be selected because the final key comes last.
"""
from __future__ import annotations

import re

_CONCLUSION = re.compile(r'"Conclusion"\s*:\s*"(pos|neg)"', re.IGNORECASE)


def extract_conclusion(text: str | None) -> str | None:
    hits = _CONCLUSION.findall(text or "")
    return hits[-1].lower() if hits else None
