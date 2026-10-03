"""Typed, source-backed semantic assertions, kept distinct from extraction guesses."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator

EntityKind = Literal["Problem", "Method", "Preprocessing", "Condition", "Tool", "Model", "Dataset", "Metric", "Finding", "Parameter", "PolicyInstrument", "KnowledgeUnit"]
RelationKind = Literal["solves", "requires", "uses_preprocessing", "uses_model", "implemented_by", "alternative_to",
                       "evaluated_on", "has_limitation", "reports_result", "works_by", "supports", "has_tradeoff",
                       "uses_parameter", "uses_metric", "depends_on", "has_property", "uses_dataset", "uses_method", "has_scope", "mitigates"]


class SemanticEntity(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(min_length=2, max_length=900)
    type: EntityKind
    aliases: list[str] = Field(default_factory=list)


class SourceQuote(BaseModel):
    model_config = ConfigDict(extra="allow")
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_source_context(self):
        if not self.quote.strip():
            raise ValueError("Source evidence cannot be whitespace only")
        if len(self.quote) < 8:
            extra = self.model_extra or {}
            if extra.get("evidence_type") != "table_cell":
                raise ValueError("Short evidence requires reviewed table-cell context; extend narrative/algorithm quotes with original context")
            if not all(extra.get(key) for key in ["table_id", "row_label", "column_label"]):
                raise ValueError("Short table evidence requires table, row and column coordinates")
        return self


class SemanticAssertion(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    source: SemanticEntity
    relation: RelationKind
    target: SemanticEntity
    text: str = Field(min_length=12)
    conditions: list[str] = Field(default_factory=list)
    evidence: list[SourceQuote] = Field(min_length=1)
