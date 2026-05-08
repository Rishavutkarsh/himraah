from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from himraah_dataset.generator import build_dataset
from himraah_dataset.validators import validate_dataset


TEST_ROOT = Path(__file__).resolve().parents[1] / "test_runs"


def test_generated_dataset_passes_validation_but_blocks_sft():
    out_dir = TEST_ROOT / "starter_gate"
    build_dataset(out_dir)
    result = validate_dataset(out_dir)
    assert result["valid"], result["errors"]
    report = json.loads((out_dir / "review_report.json").read_text(encoding="utf-8"))
    assert report["gate_status"] == "BLOCKED_PENDING_THREE_REVIEWS"
    assert report["sft_allowed"] is False


def test_eval_has_route_awareness_labels():
    out_dir = TEST_ROOT / "starter_eval"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [
        json.loads(line)
        for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(eval_rows) == 80
    for row in eval_rows:
        assert "expected_route_facts" in row
        assert "forbidden_route_claims" in row
        assert "required_safety_actions" in row
        assert "required_uncertainty_notes" in row
        assert "acceptable_hinglish_terms" in row


def test_high_risk_examples_have_escalation_and_what_not_to_do():
    out_dir = TEST_ROOT / "starter_risk"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    rows = []
    for name in ["sft_text.jsonl", "sft_vision.jsonl"]:
        rows.extend(json.loads(line) for line in (out_dir / name).read_text(encoding="utf-8").splitlines() if line.strip())
    risky = [row for row in rows if row["category"] in {"safety_high_risk", "safety_urgent"}]
    assert risky
    assert all(row["assistant_response"]["what_not_to_do"] for row in risky)
    assert all(row["assistant_response"]["escalation_signs"] for row in risky)


def test_fact_sources_are_present_on_examples():
    out_dir = TEST_ROOT / "starter_sources"
    build_dataset(out_dir)
    result = validate_dataset(out_dir)
    assert result["valid"], result["errors"]


def test_non_safety_eval_rows_have_companion_labels():
    out_dir = TEST_ROOT / "starter_eval_labels"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [
        json.loads(line)
        for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_safety = [row for row in eval_rows if row["safety_actions_required"] is False]
    assert non_safety
    assert all(row["required_companion_behaviors"] for row in non_safety)


def test_eval_runner_and_scorer_require_all_systems():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_eval_harness"
    pred_file = TEST_ROOT / "predictions.jsonl"
    build_dataset(out_dir)
    result = validate_dataset(out_dir)
    assert result["valid"], result["errors"]

    subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "run_eval.py"),
            "--dataset-dir",
            str(out_dir),
            "--out-file",
            str(pred_file),
            "--backend",
            "mock",
            "--allow-mock",
        ],
        check=True,
    )
    rows = [json.loads(line) for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_ids = {json.loads(line)["eval_id"] for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    systems = {"base_zero_shot", "base_with_himraah_context", "fine_tuned_himraah_context"}
    assert len(rows) == len(eval_ids) * len(systems)
    assert {row["system"] for row in rows} == systems

    subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "score_eval.py"),
            "--eval-file",
            str(out_dir / "eval.jsonl"),
            "--predictions-file",
            str(pred_file),
        ],
        check=True,
    )


def test_scorer_fails_when_prediction_coverage_missing():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_eval_missing"
    pred_file = TEST_ROOT / "predictions_missing.jsonl"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [
        json.loads(line)
        for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pred_file.write_text(
        json.dumps({"eval_id": eval_rows[0]["eval_id"], "system": "base_zero_shot", "prediction": "partial"}) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "score_eval.py"),
            "--eval-file",
            str(out_dir / "eval.jsonl"),
            "--predictions-file",
            str(pred_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing predictions" in (result.stderr + result.stdout)


def test_validator_requires_all_three_named_reviewers():
    out_dir = TEST_ROOT / "starter_bad_reviewers"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    report_path = out_dir / "review_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["reviewers"] = {"safety": {"status": "APPROVE", "required_before_sft": True}}
    report["gate_status"] = "APPROVED_FOR_SFT"
    report["sft_allowed"] = True
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_dataset(out_dir)
    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated["gate_status"] == "BLOCKED_PENDING_THREE_REVIEWS"
    assert updated["sft_allowed"] is False


def approve_report(out_dir: Path) -> None:
    validate_dataset(out_dir)
    report_path = out_dir / "review_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for reviewer in ["safety", "source_grounding", "training_eval"]:
        report["reviewers"][reviewer] = {"status": "APPROVE", "required_before_sft": True}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_dataset(out_dir)


def test_gemma_export_refuses_unapproved_dataset():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_export_blocked"
    export_dir = TEST_ROOT / "export_blocked"
    build_dataset(out_dir)
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_gemma_sft_dataset.py"),
            "--dataset-dir",
            str(out_dir),
            "--out-dir",
            str(export_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not approved for SFT" in (result.stderr + result.stdout)


def test_gemma_export_emits_required_fields_and_manifest():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_export_ok"
    export_dir = TEST_ROOT / "export_ok"
    build_dataset(out_dir)
    approve_report(out_dir)
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_gemma_sft_dataset.py"),
            "--dataset-dir",
            str(out_dir),
            "--out-dir",
            str(export_dir),
        ],
        check=True,
    )
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["project"] == "himraah"
    assert manifest["dataset_name"] == "himraah-text-sft-approved"
    row = json.loads((export_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for field in ["id", "prompt", "response", "text", "messages", "source_ids", "fact_ids", "category", "route_segment"]:
        assert field in row
    assert row["prompt"].startswith("<start_of_turn>user")
    assert "<start_of_turn>model" in row["text"]
    assert (export_dir / "eval_prompts.jsonl").exists()
    assert (export_dir / "eval_rubric.jsonl").exists()


def test_preflight_rejects_wrong_project_manifest():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = TEST_ROOT / "wrong_manifest"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.jsonl", "eval.jsonl", "eval_prompts.jsonl", "eval_rubric.jsonl", "training_config.json"]:
        (data_dir / name).write_text("{}\n" if name.endswith(".jsonl") else "{}", encoding="utf-8")
    (data_dir / "manifest.json").write_text(json.dumps({"project": "wrong", "dataset_name": "wrong"}), encoding="utf-8")
    env = dict(**__import__("os").environ, HIMRAAH_DATA_DIR=str(data_dir), HIMRAAH_OUT_DIR=str(TEST_ROOT / "preflight_out"))
    result = subprocess.run(
        [sys.executable, str(project_root / "kaggle_gemma_preflight" / "gemma_lora_preflight.py")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "wrong dataset manifest" in (result.stderr + result.stdout)


def test_himraah_production_scripts_have_no_sankat_coupling():
    project_root = Path(__file__).resolve().parents[1]
    forbidden = ["SANKAT" + "_", "sankat" + "-saathi"]
    roots = ["scripts", "training", "evals", "kaggle_gemma_preflight", "kaggle_gemma_sft", "kaggle_gemma_eval", "src"]
    offenders = []
    for root in roots:
        for path in (project_root / root).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".toml"}:
                text = path.read_text(encoding="utf-8")
                if any(term in text for term in forbidden):
                    offenders.append(str(path.relative_to(project_root)))
    assert offenders == []


def test_scorer_fails_global_current_claims():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_global_claim"
    pred_file = TEST_ROOT / "predictions_global_claim.jsonl"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [json.loads(line) for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for system in ["base_zero_shot", "base_with_himraah_context", "fine_tuned_himraah_context"]:
        for row in eval_rows:
            prediction = "The weather is clear and rescue is available within 20 minutes. " + " ".join(row["acceptable_hinglish_terms"])
            rows.append({"eval_id": row["eval_id"], "system": system, "prediction": prediction})
    pred_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "score_eval.py"),
            "--eval-file",
            str(out_dir / "eval.jsonl"),
            "--predictions-file",
            str(pred_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["base_zero_shot"]["pass_rate"] < 1.0
    assert "primary_delta_pass_rate" in payload


def test_transformers_eval_requires_adapter_dir():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_adapter_required"
    pred_file = TEST_ROOT / "predictions_adapter_required.jsonl"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "run_eval.py"),
            "--dataset-dir",
            str(out_dir),
            "--out-file",
            str(pred_file),
            "--backend",
            "transformers",
            "--base-model",
            "dummy-base",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--adapter-dir is required" in (result.stderr + result.stdout)


def passing_prediction(row: dict, extra: str = "") -> str:
    if row.get("safety_actions_required"):
        safety = "; ".join(row["required_safety_actions"][:3])
        uncertainty = "; ".join(row["required_uncertainty_notes"][:2])
        hinglish = "; ".join(row["acceptable_hinglish_terms"][:2])
        return f"Risk: caution/high. Do: {safety}. Note: {uncertainty}. Hinglish: {hinglish}. {extra}"
    behavior = "; ".join(row["required_companion_behaviors"][:2])
    hinglish = "; ".join(row["acceptable_hinglish_terms"][:2])
    return f"Companion answer: {behavior}. Hinglish terms: {hinglish}. I will not invent current route facts. {extra}"


def write_full_predictions(path: Path, eval_rows: list[dict], *, reportable=True, backend="kaggle_transformers_peft", extra: str = "") -> None:
    rows = []
    for system in ["base_zero_shot", "base_with_himraah_context", "fine_tuned_himraah_context"]:
        for row in eval_rows:
            pred = {
                "eval_id": row["eval_id"],
                "system": system,
                "prediction": passing_prediction(row, extra=extra),
                "backend": backend,
                "model": "test-model",
            }
            if reportable != "missing":
                pred["reportable"] = reportable
            rows.append(pred)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_scorer_require_reportable_rejects_missing_false_and_string_values():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_reportable_contract"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [json.loads(line) for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for value in ["missing", False, "true"]:
        pred_file = TEST_ROOT / f"predictions_reportable_{value}.jsonl"
        write_full_predictions(pred_file, eval_rows, reportable=value)
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "evals" / "score_eval.py"),
                "--eval-file",
                str(out_dir / "eval.jsonl"),
                "--predictions-file",
                str(pred_file),
                "--require-reportable",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        payload = json.loads(result.stdout)
        assert payload["reportable_failures"]


def test_scorer_require_reportable_rejects_mock_backend():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_reportable_mock"
    pred_file = TEST_ROOT / "predictions_reportable_mock.jsonl"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [json.loads(line) for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    write_full_predictions(pred_file, eval_rows, reportable=True, backend="mock")
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "score_eval.py"),
            "--eval-file",
            str(out_dir / "eval.jsonl"),
            "--predictions-file",
            str(pred_file),
            "--require-reportable",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["mock_failures"]


def test_scorer_flags_paraphrased_current_claims_under_strict_scoring():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_current_claim_regex"
    pred_file = TEST_ROOT / "predictions_current_claim_regex.jsonl"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [json.loads(line) for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    write_full_predictions(pred_file, eval_rows, reportable=True, extra="The route is currently open today and Jio works everywhere.")
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "score_eval.py"),
            "--eval-file",
            str(out_dir / "eval.jsonl"),
            "--predictions-file",
            str(pred_file),
            "--require-reportable",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["current_claim_failures"]


def test_scorer_allows_safe_current_status_disclaimers():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_current_claim_disclaimer"
    pred_file = TEST_ROOT / "predictions_current_claim_disclaimer.jsonl"
    build_dataset(out_dir)
    validate_dataset(out_dir)
    eval_rows = [json.loads(line) for line in (out_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    write_full_predictions(pred_file, eval_rows, reportable=True, extra="I cannot confirm current route status or provide live weather; prepare offline.")
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "evals" / "score_eval.py"),
            "--eval-file",
            str(out_dir / "eval.jsonl"),
            "--predictions-file",
            str(pred_file),
            "--require-reportable",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["current_claim_failures"] == []


def test_kaggle_scripts_include_hardened_reportable_and_4bit_paths():
    project_root = Path(__file__).resolve().parents[1]
    preflight = (project_root / "kaggle_gemma_preflight" / "gemma_lora_preflight.py").read_text(encoding="utf-8")
    sft = (project_root / "kaggle_gemma_sft" / "gemma_lora_sft.py").read_text(encoding="utf-8")
    eval_script = (project_root / "kaggle_gemma_eval" / "gemma_lora_eval.py").read_text(encoding="utf-8")
    assert "bitsandbytes" in preflight
    assert "AutoModelForCausalLM.from_pretrained" in preflight
    assert "max_new_tokens=2" in preflight
    assert "BitsAndBytesConfig" in eval_script
    assert "--require-reportable" in eval_script
    assert "eval_score.json" in eval_script
    assert "trainable_params <= 0" in sft
    assert "implausibly high" in sft
    combined = preflight + sft + eval_script
    assert "SANKAT" + "_" not in combined
    assert "sankat" + "-saathi" not in combined


def test_readme_documents_current_himraah_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    for required in [
        "prepare_gemma_sft_dataset.py",
        "kaggle_gemma_preflight",
        "kaggle_gemma_sft",
        "kaggle_gemma_eval",
        "--require-reportable",
    ]:
        assert required in readme
    for keyword in ["kaggle", "smoke", "reportable", "no local heavy eval", "no local parallel processing"]:
        assert keyword in lowered


def test_gemma_export_manifest_includes_reproducibility_metadata():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = TEST_ROOT / "starter_export_repro"
    export_dir = TEST_ROOT / "export_repro"
    build_dataset(out_dir)
    approve_report(out_dir)
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "prepare_gemma_sft_dataset.py"),
            "--dataset-dir",
            str(out_dir),
            "--out-dir",
            str(export_dir),
        ],
        check=True,
    )
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["project"] == "himraah"
    assert manifest["exported_at_utc"]
    assert manifest["git_commit"]
    assert manifest["source_dataset_hashes"]["sft_text.jsonl"]
    assert manifest["training_config_sha256"]
    assert manifest["export_bundle_sha256"]
    listed = {item["path"] for item in manifest["export_file_manifest"]}
    assert {"train.jsonl", "eval.jsonl", "eval_prompts.jsonl", "eval_rubric.jsonl", "training_config.json"} <= listed
    assert "manifest.json" not in listed
    assert manifest["git_commit"]


def test_kaggle_scripts_include_smoke_sft_and_environment_capture():
    project_root = Path(__file__).resolve().parents[1]
    preflight = (project_root / "kaggle_gemma_preflight" / "gemma_lora_preflight.py").read_text(encoding="utf-8")
    sft = (project_root / "kaggle_gemma_sft" / "gemma_lora_sft.py").read_text(encoding="utf-8")
    eval_script = (project_root / "kaggle_gemma_eval" / "gemma_lora_eval.py").read_text(encoding="utf-8")
    for text in [preflight, sft, eval_script]:
        assert "nvidia-smi" in text
        assert "disk_report" in text
        assert "resolved_paths" in text
    assert "HIMRAAH_SMOKE_STEPS" in sft
    assert "reload_adapter_smoke" in sft
    assert "MAX_LORA_TRAINABLE_PCT" in sft
    assert "HIMRAAH_EVAL_MAX_NEW_TOKENS" in eval_script


def test_kaggle_scripts_write_failure_artifacts_and_quality_caveats():
    project_root = Path(__file__).resolve().parents[1]
    scripts = {
        "preflight": (project_root / "kaggle_gemma_preflight" / "gemma_lora_preflight.py").read_text(encoding="utf-8"),
        "sft": (project_root / "kaggle_gemma_sft" / "gemma_lora_sft.py").read_text(encoding="utf-8"),
        "eval": (project_root / "kaggle_gemma_eval" / "gemma_lora_eval.py").read_text(encoding="utf-8"),
    }
    assert "preflight_failure.json" in scripts["preflight"]
    assert "sft_failure.json" in scripts["sft"]
    assert "eval_failure.json" in scripts["eval"]
    for text in scripts.values():
        assert "failure_taxonomy" in text
        assert "last_completed_checkpoint" in text
        assert "mounted_inputs" in text
    assert "quality_inference" in scripts["preflight"]
    assert "quality_inference" in scripts["sft"]
    assert "HIMRAAH_MIN_PRIMARY_DELTA" in scripts["eval"]
    assert "base_with_himraah_context" in scripts["eval"]


def test_himraah_docs_capture_kaggle_decision_and_fallback_policy():
    project_root = Path(__file__).resolve().parents[1]
    runbook = (project_root / "docs" / "kaggle_runbook.md").read_text(encoding="utf-8").lower()
    failure = (project_root / "docs" / "failure_analysis_template.md").read_text(encoding="utf-8").lower()
    submission = (project_root / "docs" / "submission_skeleton.md").read_text(encoding="utf-8").lower()
    assert "himraah_smoke_steps=1" in runbook
    assert "base_with_himraah_context" in runbook
    assert "zero blocker failures" in runbook
    assert "rollback" in runbook or "fallback" in runbook
    assert "non-evaluative" in runbook
    assert "immutable kaggle slugs" in runbook
    assert "himraah_min_primary_delta" in runbook
    assert "use only scripts under `himraah/kaggle_*`" in runbook
    assert "prepare_kaggle_preflight_package.py" in runbook
    assert "rishavutkarsh/himraah-text-sft-approved" in runbook
    assert "blocker" in failure
    assert "dataset patch manifest" in failure
    assert "dpo remains blocked" in failure
    assert "regression against fallback" in failure
    assert "reviewer/sign-off" in failure
    assert "offline route-pack architecture" in submission
    assert "base-with-context" in submission
    assert "future extension" in submission
    assert "risk register" in submission
    assert "concrete gangotri" in submission


def test_preflight_package_script_enforces_expected_kaggle_sources():
    project_root = Path(__file__).resolve().parents[1]
    script = (project_root / "scripts" / "prepare_kaggle_preflight_package.py").read_text(encoding="utf-8")
    assert "rishavutkarsh/himraah-text-sft-approved" in script
    assert "rishavutkarsh/himraah-gemma-e2b-preflight" in script
    assert "rishavutkarsh/himraah-transformers-wheels" in script
    assert "google/gemma-4/Transformers/gemma-4-e2b-it/1" in script
    assert "assert_clean_repo" in script
    assert "scan_export" in script
    assert "dataset-metadata.json" in script
    assert "Preflight tiny generation is a plumbing canary only" in script
