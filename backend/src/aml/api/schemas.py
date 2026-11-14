from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from aml.detect.common import Typology


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlertEdge(Strict):
    tx_id: int
    src: str
    dst: str
    ts: datetime
    amount: float = Field(ge=0)
    currency: str


class TimeSpan(Strict):
    start: datetime
    end: datetime


class Contribution(Strict):
    name: str
    contribution: float


class Alert(Strict):
    alert_id: str
    rank: int = Field(ge=1)
    risk_score: float = Field(ge=0, le=100)
    primary_typology: Typology
    secondary_typologies: list[Typology]
    accounts: list[str]
    edges: list[AlertEdge]
    time_span: TimeSpan
    total_value: float = Field(ge=0)
    top_features: list[Contribution]
    summary: str
    summary_source: Literal["template", "llm_validated"]
    evidence: dict[str, Any]

    @model_validator(mode="after")
    def validate_evidence(self) -> "Alert":
        if not self.edges or self.time_span.start > self.time_span.end:
            raise ValueError("alert must have edges and an ordered time span")
        if len({e.tx_id for e in self.edges}) != len(self.edges):
            raise ValueError("duplicate evidence transaction")
        if any(e.src not in self.accounts or e.dst not in self.accounts or not self.time_span.start <= e.ts <= self.time_span.end for e in self.edges):
            raise ValueError("edge outside alert accounts/time span")
        if [(e.ts, e.tx_id) for e in self.edges] != sorted((e.ts, e.tx_id) for e in self.edges):
            raise ValueError("evidence must follow temporal order")
        return self
