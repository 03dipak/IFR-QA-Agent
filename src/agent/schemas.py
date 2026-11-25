from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

ToolName = Literal["search_index", "calculator"]


class PlanStep(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)


class FigureRequest(BaseModel):
    company_id: str = ""
    label: str = ""


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=10)
    requested: list[FigureRequest] = Field(default_factory=list)


class Figure(BaseModel):
    company_id: str = ""
    label: str = ""
    period_column: Literal["current", "prior", "unknown"] = "unknown"
    value: str | int | float = ""
    chunk_id: str = ""
    prior: str | int | float | None = None
    unit: str = ""
    page: int | None = None
    label_equivalence: bool = False


class Answer(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    figures: list[Figure] = Field(default_factory=list)


class Critique(BaseModel):
    verdict: Literal["accept", "revise"]
    feedback: str = ""
    next_steps: list[PlanStep] = Field(default_factory=list, max_length=10)

    @field_validator("next_steps", mode="before")
    @classmethod
    def _coerce_next_steps(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        detected = (info.context or {}).get("detected_companies") or []
        company = detected[0] if len(detected) == 1 else None
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, str):
                query = item.strip()
                if not query:
                    continue
                args: dict[str, Any] = {"query": query, "k": 5}
                if company:
                    args["company"] = company
                coerced.append({"tool": "search_index", "args": args})
            else:
                coerced.append(item)
        return coerced
