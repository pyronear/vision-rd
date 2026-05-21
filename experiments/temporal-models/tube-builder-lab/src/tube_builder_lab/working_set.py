"""Load the curated working set (targets + control) from working_set.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class WorkingItem:
    key: str
    note: str | None = None


@dataclass
class WorkingSet:
    targets: list[WorkingItem]
    control: list[WorkingItem]

    def all(self) -> list[WorkingItem]:
        return [*self.targets, *self.control]


def _items(raw: list | None) -> list[WorkingItem]:
    return [WorkingItem(key=o["key"], note=o.get("note")) for o in (raw or [])]


def load_working_set(path: Path) -> WorkingSet:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    return WorkingSet(
        targets=_items(payload.get("targets")),
        control=_items(payload.get("control")),
    )
