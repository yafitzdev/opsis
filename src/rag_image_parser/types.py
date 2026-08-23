from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class OutputKind(str, Enum):
    DESCRIPTION = "description"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class ParseResult:
    kind: OutputKind
    text: str
    latency_seconds: float
    model_id: str
    prompt_version: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data
