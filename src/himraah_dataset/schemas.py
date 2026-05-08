from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceType = Literal[
    "official_government",
    "official_advisory",
    "medical_authority",
    "conservation_research",
    "reputable_trek_operator",
    "travel_blog",
    "synthetic_grounded",
]
FactStability = Literal["stable", "seasonal_or_variable", "current_status_required"]
RiskLevel = Literal["low", "caution", "high", "critical", "cannot_determine"]
Category = Literal[
    "companion_route_qa",
    "field_guide",
    "planning",
    "language_help",
    "culture_learning",
    "safety_high_risk",
    "safety_urgent",
]
Split = Literal["train_sft", "train_dpo", "eval_seen_route_unseen_prompt", "eval_unseen_scenario"]


@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    url: str
    organization: str
    source_type: SourceType
    accessed_at: str
    notes: str


@dataclass(frozen=True)
class Fact:
    fact_id: str
    source_id: str
    source_type: SourceType
    fact_stability: FactStability
    category: str
    route_segment: str
    claim: str
    allowed_phrasings: list[str]
    forbidden_claims: list[str]
    tags: list[str]


@dataclass(frozen=True)
class Phrase:
    phrase_id: str
    category: Category
    english_intent: str
    hindi_hinglish: str
    polite_register_note: str
    source_id: str = "synthetic_phrasebook"
    source_type: SourceType = "synthetic_grounded"
    fact_stability: FactStability = "stable"


@dataclass(frozen=True)
class StructuredAnswer:
    risk_level: RiskLevel
    route_context: list[str]
    answer: str
    immediate_next_steps: list[str]
    what_not_to_do: list[str]
    escalation_signs: list[str]
    missing_info: list[str]
    confidence_note: str
    hinglish: str


@dataclass(frozen=True)
class SftExample:
    example_id: str
    split: Split
    category: Category
    route_segment: str
    source_ids: list[str]
    fact_ids: list[str]
    user_prompt: str
    assistant_response: StructuredAnswer
    image_path: str | None = None
    image_observations: list[str] = field(default_factory=list)
    reviewer_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DpoDraft:
    pair_id: str
    split: Split
    category: Category
    route_segment: str
    source_ids: list[str]
    fact_ids: list[str]
    prompt: str
    chosen: StructuredAnswer
    rejected: str
    rejection_reasons: list[str]
    target_failure_mode: str


@dataclass(frozen=True)
class EvalExample:
    eval_id: str
    split: Split
    category: Category
    route_segment: str
    source_ids: list[str]
    fact_ids: list[str]
    user_prompt: str
    expected_route_facts: list[str]
    forbidden_route_claims: list[str]
    required_safety_actions: list[str]
    safety_actions_required: bool
    required_companion_behaviors: list[str]
    required_uncertainty_notes: list[str]
    acceptable_hinglish_terms: list[str]
    image_path: str | None = None


def to_jsonable(item: Any) -> dict[str, Any]:
    return asdict(item)
