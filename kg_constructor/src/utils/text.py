from __future__ import annotations

import re

DOID_PATTERN = re.compile(r"(DOID[_:])(\d+)$", re.IGNORECASE)


def clean_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_canonical_label(text: str) -> str:
    return clean_whitespace(text).upper()


def normalize_doid(doid_raw: str) -> str:
    value = doid_raw.strip()

    if value.lower().startswith("http"):
        value = value.rsplit("/", 1)[-1]

    match = DOID_PATTERN.search(value)
    if match:
        return f"DOID:{match.group(2)}"

    if value.upper().startswith("DOID:"):
        return f"DOID:{value.split(':', 1)[1].strip()}"

    return value


def clean_disease_label(text: str) -> str:
    cleaned = clean_whitespace(text)

    if "(" in cleaned:
        cleaned = cleaned.split("(", 1)[0].strip()

    return cleaned