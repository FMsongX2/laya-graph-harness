"""Source-grounded claims and reusable knowledge cards."""
from typing import Literal
from pydantic import BaseModel, Field

ClaimKind = Literal["problem", "mechanism", "preprocessing", "applicability", "requirement", "alternative", "evaluation", "limitation"]


class EvidenceQuote(BaseModel):
    fragment_id: str = Field(min_length=1)
    quote: str = Field(min_length=8)


class ExtractedClaim(BaseModel):
    subject: str = Field(min_length=2)
    kind: ClaimKind
    text: str = Field(min_length=8, max_length=800)
    evidence: list[EvidenceQuote] = Field(min_length=1, max_length=3)


class ClaimBatch(BaseModel):
    claims: list[ExtractedClaim] = Field(max_length=12)


class CardPlan(BaseModel):
    title: str = Field(min_length=3)
    subject: str = Field(min_length=2)
    claim_ids: list[str] = Field(min_length=1, max_length=18)


class CardPlans(BaseModel):
    cards: list[CardPlan] = Field(min_length=1, max_length=4)


class ClaimAudit(BaseModel):
    claim_id: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    reason: str


class ClaimAudits(BaseModel):
    audits: list[ClaimAudit]
