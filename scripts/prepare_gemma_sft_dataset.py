from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from himraah_dataset.validators import validate_dataset


DATASET_NAME = "himraah-text-sft-approved"
DEFAULT_CONFIG = {
    "seed": 17,
    "max_length": 768,
    "num_train_epochs": 1.0,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "completion_only_loss": True,
    "model_family": "gemma-e2b",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip()


def file_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def flatten_response(answer: dict[str, Any]) -> str:
    lines = [
        f"risk_level: {answer['risk_level']}",
        "route_context:",
        *[f"- {item}" for item in answer.get("route_context", [])],
        f"answer: {answer['answer']}",
        "immediate_next_steps:",
        *[f"- {item}" for item in answer.get("immediate_next_steps", [])],
        "what_not_to_do:",
        *[f"- {item}" for item in answer.get("what_not_to_do", [])],
        "escalation_signs:",
        *[f"- {item}" for item in answer.get("escalation_signs", [])],
        "missing_info:",
        *[f"- {item}" for item in answer.get("missing_info", [])],
        f"confidence_note: {answer['confidence_note']}",
        f"hinglish: {answer['hinglish']}",
    ]
    return "\n".join(lines)


def gemma_prompt(user_prompt: str) -> str:
    return f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"


def gemma_text(user_prompt: str, response: str) -> str:
    return f"{gemma_prompt(user_prompt)}{response}<end_of_turn>"


def export_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    response = flatten_response(row["assistant_response"])
    prompt = gemma_prompt(row["user_prompt"])
    return {
        "id": row["example_id"],
        "prompt": prompt,
        "response": response,
        "text": gemma_text(row["user_prompt"], response),
        "messages": [
            {"role": "user", "content": row["user_prompt"]},
            {"role": "model", "content": response},
        ],
        "category": row["category"],
        "route_segment": row["route_segment"],
        "source_ids": row["source_ids"],
        "fact_ids": row["fact_ids"],
        "risk_level": row["assistant_response"]["risk_level"],
    }


def route_context(dataset_dir: Path) -> str:
    claims: list[str] = []
    for name in ["route_facts.jsonl", "safety_facts.jsonl", "field_guide.jsonl"]:
        for row in read_jsonl(dataset_dir / name):
            claims.append(f"- {row['claim']}")
    return "\n".join(claims[:60])


def export_eval_prompt(row: dict[str, Any], context: str) -> dict[str, Any]:
    system_context = (
        "You are HimRaah, an offline Gangotri-Chirbasa-Bhojbasa-Gomukh route companion. "
        "Use only bundled route context. Do not invent live weather, closures, prices, permits, rescue, transport, "
        "accommodation, exact carrier coverage, or current availability. Be bilingual when useful and conservative for safety."
    )
    contextual_user = f"{system_context}\n\nBundled context:\n{context}\n\nUser question: {row['user_prompt']}"
    return {
        "id": row["eval_id"],
        "eval_id": row["eval_id"],
        "prompt": gemma_prompt(row["user_prompt"]),
        "context_prompt": gemma_prompt(contextual_user),
        "messages": [{"role": "user", "content": row["user_prompt"]}],
        "context_messages": [{"role": "user", "content": contextual_user}],
        "category": row["category"],
        "route_segment": row["route_segment"],
        "source_ids": row["source_ids"],
        "fact_ids": row["fact_ids"],
    }


def assert_approved(dataset_dir: Path) -> dict[str, Any]:
    result = validate_dataset(dataset_dir, update_report=True)
    if not result["valid"]:
        raise SystemExit(f"BLOCKED: HimRaah dataset validation failed: {result['errors']}")
    report = result["report"]
    if report.get("gate_status") != "APPROVED_FOR_SFT" or report.get("sft_allowed") is not True:
        raise SystemExit(f"BLOCKED: dataset is not approved for SFT: gate_status={report.get('gate_status')}")
    return report


def build_export(dataset_dir: Path, out_dir: Path, eval_holdout: int) -> dict[str, Any]:
    review_report = assert_approved(dataset_dir)
    sft_rows = [export_sft_row(row) for row in read_jsonl(dataset_dir / "sft_text.jsonl")]
    if len(sft_rows) <= eval_holdout:
        raise SystemExit("BLOCKED: eval holdout must be smaller than text SFT row count")
    train_rows = sft_rows[:-eval_holdout]
    loss_eval_rows = sft_rows[-eval_holdout:]
    rubric_rows = read_jsonl(dataset_dir / "eval.jsonl")
    context = route_context(dataset_dir)
    eval_prompt_rows = [export_eval_prompt(row, context) for row in rubric_rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "eval.jsonl", loss_eval_rows)
    write_jsonl(out_dir / "eval_prompts.jsonl", eval_prompt_rows)
    write_jsonl(out_dir / "eval_rubric.jsonl", rubric_rows)

    training_config = dict(DEFAULT_CONFIG)
    (out_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")

    manifest = {
        "project": "himraah",
        "dataset_name": DATASET_NAME,
        "route": review_report["route"],
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_dataset_dir": str(dataset_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(loss_eval_rows),
        "eval_prompt_rows": len(eval_prompt_rows),
        "train_ids": [row["id"] for row in train_rows],
        "eval_ids": [row["id"] for row in loss_eval_rows],
        "rubric_eval_ids": [row["eval_id"] for row in rubric_rows],
        "review_report_snapshot": review_report,
        "source_dataset_hashes": {
            name: sha256_file(dataset_dir / name)
            for name in [
                "sources_manifest.jsonl",
                "route_facts.jsonl",
                "safety_facts.jsonl",
                "field_guide.jsonl",
                "phrasebook.jsonl",
                "sft_text.jsonl",
                "eval.jsonl",
                "review_report.json",
            ]
        },
        "training_config_sha256": sha256_file(out_dir / "training_config.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["export_file_manifest"] = file_manifest(out_dir)
    manifest["export_bundle_sha256"] = hashlib.sha256(
        "".join(item["sha256"] for item in manifest["export_file_manifest"]).encode("utf-8")
    ).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an approved HimRaah Gemma SFT Kaggle dataset bundle.")
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "processed" / "starter"))
    parser.add_argument("--out-dir", default=str(ROOT / "exports" / DATASET_NAME))
    parser.add_argument("--eval-holdout", type=int, default=25)
    args = parser.parse_args()
    manifest = build_export(Path(args.dataset_dir), Path(args.out_dir), args.eval_holdout)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
