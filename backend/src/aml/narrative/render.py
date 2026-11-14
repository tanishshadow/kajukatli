"""Validate placeholder narratives and render only evidence-backed fields."""
import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class Sentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentences: list[Sentence] = Field(min_length=1)


class OfflineLLMClient:
    def generate(self, evidence: dict[str, Any], guidance: str) -> dict[str, Any]:
        return {"sentences": [{"text": "The transaction sequence shows patterns consistent with {T1.currency} transfers.",
                              "evidence_ids": ["T1"]}]}


PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def render_narrative(payload: dict[str, Any], evidence: dict[str, Any]) -> str:
    narrative = Narrative.model_validate(payload)
    allowed = set(evidence.get("entities", {})) | set(evidence.get("transactions", {}))
    rendered: list[str] = []
    previous_order = ""
    for sentence in narrative.sentences:
        if any(identifier not in allowed for identifier in sentence.evidence_ids):
            raise ValueError("unknown evidence id")
        for identifier in sentence.evidence_ids:
            if identifier not in sentence.text:
                raise ValueError("evidence reference must appear as a placeholder")
        if re.search(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?", PLACEHOLDER.sub("", sentence.text)):
            raise ValueError("literal numeric facts are forbidden; use evidence placeholders")
        placeholders = [match.group(1) for match in PLACEHOLDER.finditer(sentence.text)]
        for key in placeholders:
            tx_id, _, field = key.partition(".")
            if key not in evidence.get("entities", {}) and not (
                tx_id in evidence.get("transactions", {}) and field in evidence["transactions"][tx_id]):
                raise ValueError("unresolved or unsupported placeholder")
        orders = [str(evidence["transactions"][key.partition('.')[0]].get("ts", ""))
                  for key in placeholders if key.partition('.')[0] in evidence.get("transactions", {})]
        if orders and min(orders) < previous_order:
            raise ValueError("event order is not chronological")
        if orders:
            previous_order = max(orders)
        def substitute(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in evidence.get("entities", {}):
                return str(evidence["entities"][key])
            tx_id, _, field = key.partition(".")
            if tx_id in evidence.get("transactions", {}) and field in evidence["transactions"][tx_id]:
                return str(evidence["transactions"][tx_id][field])
            raise ValueError("unresolved or unsupported placeholder")
        value = PLACEHOLDER.sub(substitute, sentence.text)
        if PLACEHOLDER.search(value):
            raise ValueError("unresolved placeholder")
        rendered.append(value)
    return " ".join(rendered)
