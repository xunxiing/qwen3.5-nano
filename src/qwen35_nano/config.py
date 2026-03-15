from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(raw=yaml.safe_load(handle))

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

