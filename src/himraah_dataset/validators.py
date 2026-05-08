from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "sources_manifest.jsonl",
    "route_facts.jsonl",
    "safety_facts.jsonl",
    "field_guide.jsonl",
    "phrasebook.jsonl",
    "sft_text.jsonl",
    "sft_vision.jsonl",
    "dpo_draft.jsonl",
    "eval.jsonl",
    "review_report.json",
]
REQUIRED_REVIEWERS = {"safety", "source_grounding", "training_eval"}

COMPANION_CATEGORIES = {"companion_route_qa", "field_guide", "planning", "language_help", "culture_learning"}
HIGH_RISK_CATEGORIES = {"safety_high_risk", "safety_urgent"}
CURRENT_CLAIM_PHRASES = [
    "weather is",
    "route is open",
    "route is closed",
    "rescue is available",
    "correct mule price",
    "correct porter price",
    "today is rs",
    "jio will",
    "airtel will",
    "bsnl will",
    "definitely no network",
    "guaranteed network",
]
VISION_UNSAFE_PHRASES = [
    "safe to eat",
    "safe to touch",
    "campsite is safe",
    "glacier edge looks stable",
    "drink untreated",
]


def canonical_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    for marker in [" variation ", " scenario ", " held-out case ", " image case "]:
        if marker in lowered:
            lowered = lowered.split(marker)[0]
    digits_removed = "".join(ch for ch in lowered if not ch.isdigit())
    return " ".join(digits_removed.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno} is not valid JSON: {exc}") from exc
    return rows


def text_blob(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False).lower()


def validate_dataset(dataset_dir: Path, update_report: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED_FILES:
        if not (dataset_dir / name).exists():
            errors.append(f"missing required file: {name}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    sources = read_jsonl(dataset_dir / "sources_manifest.jsonl")
    source_ids = {row["source_id"] for row in sources}
    facts = [
        *read_jsonl(dataset_dir / "route_facts.jsonl"),
        *read_jsonl(dataset_dir / "safety_facts.jsonl"),
        *read_jsonl(dataset_dir / "field_guide.jsonl"),
    ]
    fact_ids = {row["fact_id"] for row in facts}
    fact_source_by_id = {row["fact_id"]: row["source_id"] for row in facts}
    text_examples = read_jsonl(dataset_dir / "sft_text.jsonl")
    vision_examples = read_jsonl(dataset_dir / "sft_vision.jsonl")
    dpo_drafts = read_jsonl(dataset_dir / "dpo_draft.jsonl")
    eval_examples = read_jsonl(dataset_dir / "eval.jsonl")

    for fact in facts:
        for field in ["source_id", "source_type", "fact_stability"]:
            if not fact.get(field):
                errors.append(f"fact {fact.get('fact_id', '<missing>')} missing {field}")
        if fact.get("source_id") not in source_ids:
            errors.append(f"fact {fact.get('fact_id')} references unknown source {fact.get('source_id')}")
        if fact.get("fact_stability") == "current_status_required":
            warnings.append(f"fact {fact.get('fact_id')} requires current status; ensure examples do not state it as timeless")

    all_train_prompts = {row["user_prompt"].strip().lower() for row in [*text_examples, *vision_examples]}
    for row in eval_examples:
        prompt = row["user_prompt"].strip().lower()
        if prompt in all_train_prompts:
            errors.append(f"eval prompt duplicates train prompt: {row['eval_id']}")
        for field in [
            "expected_route_facts",
            "forbidden_route_claims",
            "required_safety_actions",
            "safety_actions_required",
            "required_companion_behaviors",
            "required_uncertainty_notes",
            "acceptable_hinglish_terms",
        ]:
            if field not in row:
                errors.append(f"eval {row.get('eval_id')} missing {field}")
        if row.get("safety_actions_required") is True and not row.get("required_safety_actions"):
            errors.append(f"eval {row.get('eval_id')} requires safety actions but list is empty")
        if row.get("safety_actions_required") is False and not row.get("required_companion_behaviors"):
            errors.append(f"eval {row.get('eval_id')} has no safety actions and no companion behavior labels")

    category_counts: dict[str, int] = {}
    for row in [*text_examples, *vision_examples]:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
        for sid in row.get("source_ids", []):
            if sid not in source_ids:
                errors.append(f"example {row.get('example_id')} references unknown source {sid}")
        for fid in row.get("fact_ids", []):
            if fid not in fact_ids and not fid.startswith("phrase_"):
                errors.append(f"example {row.get('example_id')} references unknown fact {fid}")
            expected_source = "synthetic_phrasebook" if fid.startswith("phrase_") else fact_source_by_id.get(fid)
            if expected_source and expected_source not in row.get("source_ids", []):
                errors.append(f"example {row.get('example_id')} fact {fid} requires missing source {expected_source}")
        response = row.get("assistant_response", {})
        if row["category"] in HIGH_RISK_CATEGORIES:
            if not response.get("what_not_to_do"):
                errors.append(f"high-risk example {row.get('example_id')} missing what_not_to_do")
            if not response.get("escalation_signs"):
                errors.append(f"high-risk example {row.get('example_id')} missing escalation_signs")
        blob = text_blob(row)
        for phrase in CURRENT_CLAIM_PHRASES:
            if phrase in blob and "do not" not in blob and "avoid" not in blob:
                errors.append(f"example {row.get('example_id')} may contain forbidden current claim phrase: {phrase}")

    total_sft = len(text_examples) + len(vision_examples)
    companion_count = sum(count for category, count in category_counts.items() if category in COMPANION_CATEGORIES)
    companion_ratio = companion_count / total_sft if total_sft else 0.0
    if companion_ratio < 0.40:
        errors.append(f"non-emergency companion ratio {companion_ratio:.2%} is below 40%")

    if len(text_examples) < 240:
        errors.append("sft_text count below target floor of 240")
    if not 60 <= len(vision_examples) <= 100:
        errors.append("sft_vision count must be 60-100")
    if len(eval_examples) < 80:
        errors.append("eval count below target floor of 80")

    canonical_train = {canonical_prompt(row["user_prompt"]) for row in [*text_examples, *vision_examples]}
    canonical_eval = {canonical_prompt(row["user_prompt"]) for row in eval_examples}
    if len(canonical_train) < 50:
        errors.append(f"canonical train prompt diversity too low: {len(canonical_train)} < 50")
    if len(canonical_eval) < 30:
        errors.append(f"canonical eval prompt diversity too low: {len(canonical_eval)} < 30")

    for row in vision_examples:
        blob = text_blob(row)
        if "cannot confirm" not in blob and "not proof of safety" not in blob:
            errors.append(f"vision example {row.get('example_id')} missing image uncertainty language")
        for phrase in VISION_UNSAFE_PHRASES:
            if phrase in blob and f"do not {phrase}" not in blob and "cannot confirm" not in blob:
                errors.append(f"vision example {row.get('example_id')} may contain unsafe vision phrase: {phrase}")

    for row in dpo_drafts:
        if row.get("split") != "train_dpo":
            errors.append(f"dpo draft {row.get('pair_id')} has wrong split")
        if not row.get("rejection_reasons"):
            errors.append(f"dpo draft {row.get('pair_id')} missing rejection reasons")
        for sid in row.get("source_ids", []):
            if sid not in source_ids:
                errors.append(f"dpo draft {row.get('pair_id')} references unknown source {sid}")
        for fid in row.get("fact_ids", []):
            if fid not in fact_ids and not fid.startswith("phrase_"):
                errors.append(f"dpo draft {row.get('pair_id')} references unknown fact {fid}")
            expected_source = "synthetic_phrasebook" if fid.startswith("phrase_") else fact_source_by_id.get(fid)
            if expected_source and expected_source not in row.get("source_ids", []):
                errors.append(f"dpo draft {row.get('pair_id')} fact {fid} requires missing source {expected_source}")

    for row in eval_examples:
        for sid in row.get("source_ids", []):
            if sid not in source_ids:
                errors.append(f"eval {row.get('eval_id')} references unknown source {sid}")
        for fid in row.get("fact_ids", []):
            if fid not in fact_ids and not fid.startswith("phrase_"):
                errors.append(f"eval {row.get('eval_id')} references unknown fact {fid}")
            expected_source = "synthetic_phrasebook" if fid.startswith("phrase_") else fact_source_by_id.get(fid)
            if expected_source and expected_source not in row.get("source_ids", []):
                errors.append(f"eval {row.get('eval_id')} fact {fid} requires missing source {expected_source}")

    report_path = dataset_dir / "review_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["generated_dataset_valid"] = not errors
    report["validation"] = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "category_counts": category_counts,
        "non_emergency_companion_ratio": round(companion_ratio, 4),
    }
    if errors:
        report["gate_status"] = "BLOCKED_VALIDATION_FAILED"
        report["sft_allowed"] = False
    else:
        reviewers = report.get("reviewers", {})
        reviewer_statuses = {name: reviewers.get(name, {}).get("status") for name in REQUIRED_REVIEWERS}
        if all(status == "APPROVE" for status in reviewer_statuses.values()):
            report["gate_status"] = "APPROVED_FOR_SFT"
            report["sft_allowed"] = True
        else:
            report["gate_status"] = "BLOCKED_PENDING_THREE_REVIEWS"
            report["sft_allowed"] = False
    if update_report:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {"valid": not errors, "errors": errors, "warnings": warnings, "report": report}
