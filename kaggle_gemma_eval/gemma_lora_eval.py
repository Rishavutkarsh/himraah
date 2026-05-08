"""Kaggle real-model eval for HimRaah base vs context vs LoRA adapter."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


DATASET_NAME = "himraah-text-sft-approved"
DATA_DIR = Path(os.environ.get("HIMRAAH_DATA_DIR", "/kaggle/input/himraah-text-sft-approved"))
OUT_DIR = Path(os.environ.get("HIMRAAH_OUT_DIR", "/kaggle/working/himraah_gemma_e2b_eval"))
ADAPTER_DIR = Path(os.environ.get("HIMRAAH_ADAPTER_DIR", "/kaggle/input/himraah-gemma-e2b-lora-sft/adapter"))
SYSTEMS = ["base_zero_shot", "base_with_himraah_context", "fine_tuned_himraah_context"]
USE_4BIT = os.environ.get("HIMRAAH_EVAL_PRECISION", "4bit").lower() != "fp16"
MAX_NEW_TOKENS = int(os.environ.get("HIMRAAH_EVAL_MAX_NEW_TOKENS", "384"))
MIN_PRIMARY_DELTA = float(os.environ.get("HIMRAAH_MIN_PRIMARY_DELTA", "0.01"))
LAST_CHECKPOINT = "start"


def install_transformers_wheel() -> str:
    input_root = Path("/kaggle/input")
    transformer_wheels = list(input_root.rglob("transformers-5.8.0-py3-none-any.whl")) if input_root.exists() else []
    if transformer_wheels:
        wheel_dir = transformer_wheels[0].parent
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                str(wheel_dir / "huggingface_hub-1.14.0-py3-none-any.whl"),
                str(wheel_dir / "transformers-5.8.0-py3-none-any.whl"),
            ]
        )
        return str(wheel_dir)
    return "wheel_not_found"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": command}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "command": command,
    }


def disk_report(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def package_versions() -> dict:
    versions = {}
    for name in ["torch", "transformers", "peft", "accelerate", "tokenizers", "bitsandbytes"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable:{exc}"
    return versions


def mounted_inputs() -> list[str]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    return sorted(str(path) for path in root.iterdir())


def failure_taxonomy(exc: BaseException) -> str:
    text = str(exc).lower()
    if "scoring" in text or "eval_score" in text or "reportable" in text:
        return "strict_scoring"
    if "adapter" in text or "lora" in text:
        return "adapter_input"
    if "cuda" in text or "gpu" in text or "memory" in text:
        return "gpu_or_memory"
    if "manifest" in text or "dataset" in text:
        return "dataset_manifest"
    if "bitsandbytes" in text or "transformers" in text or "package" in text or "wheel" in text:
        return "package_or_wheel"
    if "model" in text or "gemma" in text or "mounted" in text:
        return "model_mount"
    return "script_bug_or_unknown"


def write_failure(stage: str, exc: BaseException) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "himraah",
        "stage": stage,
        "ok": False,
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "exception_type": exc.__class__.__name__,
        "exception": str(exc),
        "taxonomy": failure_taxonomy(exc),
        "last_completed_checkpoint": LAST_CHECKPOINT,
        "traceback": traceback.format_exc(),
        "env": {key: value for key, value in os.environ.items() if key.startswith("HIMRAAH_")},
        "mounted_inputs": mounted_inputs(),
        "package_versions": package_versions(),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
    }
    (OUT_DIR / "eval_failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def assert_data_dir() -> dict:
    required = ["eval_prompts.jsonl", "eval_rubric.jsonl", "manifest.json"]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"missing HimRaah eval files under {DATA_DIR}: {missing}")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("project") != "himraah" or manifest.get("dataset_name") != DATASET_NAME:
        raise RuntimeError(f"wrong HimRaah eval manifest: {manifest}")
    return manifest


def find_model_path() -> Path:
    explicit = os.environ.get("HIMRAAH_MODEL_PATH")
    if explicit:
        return Path(explicit)
    candidates = [
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1",
        "/kaggle/input/models/google/gemma-4/Transformers/gemma-4-e2b-it/1",
        "/kaggle/input/gemma-4/transformers/gemma-4-e2b-it/1",
        "/kaggle/input/gemma-4/Transformers/gemma-4-e2b-it/1",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise RuntimeError("No mounted Gemma E2B-IT Transformers model found.")


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    return outputs


def quantization_report(use_4bit: bool) -> dict:
    if not use_4bit:
        return {"precision": "fp16", "load_in_4bit": False}
    return {
        "precision": "4bit",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    }


def score_predictions(eval_file: Path, predictions_file: Path, out_file: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "evals" / "score_eval.py"),
        "--eval-file",
        str(eval_file),
        "--predictions-file",
        str(predictions_file),
        "--out-file",
        str(out_file),
        "--require-reportable",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if out_file.exists():
        payload = json.loads(out_file.read_text(encoding="utf-8"))
    else:
        payload = {
            "passed": False,
            "score": 0.0,
            "hard_failures": [{"type": "scorer_crash", "stderr": result.stderr, "stdout": result.stdout}],
        }
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"HimRaah strict eval scoring failed; wrote {out_file}")
    return payload


def main() -> None:
    global LAST_CHECKPOINT
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wheel_status = install_transformers_wheel()
    LAST_CHECKPOINT = "wheel_install_checked"
    manifest = assert_data_dir()
    LAST_CHECKPOINT = "dataset_manifest_validated"
    if not ADAPTER_DIR.exists():
        raise RuntimeError(f"HimRaah adapter directory not found: {ADAPTER_DIR}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import bitsandbytes  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HimRaah Gemma eval.")
    model_path = find_model_path()
    LAST_CHECKPOINT = "model_path_resolved"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if USE_4BIT:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    base.eval()
    LAST_CHECKPOINT = "base_model_loaded"
    eval_rows = read_jsonl(DATA_DIR / "eval_prompts.jsonl")
    predictions = []
    for system in SYSTEMS:
        prompts = [row["prompt"] if system == "base_zero_shot" else row["context_prompt"] for row in eval_rows]
        if system == "fine_tuned_himraah_context":
            model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            model.eval()
            generated = generate(model, tokenizer, prompts, MAX_NEW_TOKENS)
            LAST_CHECKPOINT = "fine_tuned_predictions_generated"
        else:
            generated = generate(base, tokenizer, prompts, MAX_NEW_TOKENS)
            LAST_CHECKPOINT = f"{system}_predictions_generated"
        for row, prompt, prediction in zip(eval_rows, prompts, generated):
            predictions.append(
                {
                    "eval_id": row["eval_id"],
                    "system": system,
                    "prompt": prompt,
                    "prediction": prediction,
                    "backend": "kaggle_transformers_peft",
                    "model": str(model_path),
                    "adapter_dir": str(ADAPTER_DIR) if system == "fine_tuned_himraah_context" else "",
                    "reportable": True,
                }
            )
    out_file = OUT_DIR / "eval_predictions.jsonl"
    with out_file.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    eval_score_file = OUT_DIR / "eval_score.json"
    score_payload = score_predictions(DATA_DIR / "eval_rubric.jsonl", out_file, eval_score_file)
    LAST_CHECKPOINT = "strict_score_complete"
    acceptance = {
        "baseline": "base_with_himraah_context",
        "candidate": "fine_tuned_himraah_context",
        "minimum_primary_delta": MIN_PRIMARY_DELTA,
        "primary_delta_passed": score_payload.get("primary_delta_pass_rate", 0.0) >= MIN_PRIMARY_DELTA,
        "zero_hard_failures": not score_payload.get("hard_failures"),
        "zero_current_claim_failures": not score_payload.get("current_claim_failures"),
        "zero_reportable_failures": not score_payload.get("reportable_failures"),
        "coverage_complete": not score_payload.get("coverage_failures"),
        "fallback_rule": "Use base_with_himraah_context unless tuned LoRA is materially better and no less safe.",
        "manual_review_required": True,
    }
    acceptance["accepted_for_demo_candidate"] = all(
        [
            acceptance["primary_delta_passed"],
            acceptance["zero_hard_failures"],
            acceptance["zero_current_claim_failures"],
            acceptance["zero_reportable_failures"],
            acceptance["coverage_complete"],
        ]
    )
    report = {
        "project": "himraah",
        "stage": "strict_eval",
        "ok": True,
        "dataset_name": DATASET_NAME,
        "manifest": manifest,
        "model_path": str(model_path),
        "adapter_dir": str(ADAPTER_DIR),
        "resolved_paths": {
            "data_dir": str(DATA_DIR.resolve()),
            "out_dir": str(OUT_DIR.resolve()),
            "model_path": str(model_path.resolve()),
            "adapter_dir": str(ADAPTER_DIR.resolve()),
        },
        "systems": SYSTEMS,
        "predictions": len(predictions),
        "transformers_wheel": wheel_status,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 3),
        "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
        "quantization": quantization_report(USE_4BIT),
        "decoding": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "nvidia_smi": command_output(["nvidia-smi"]),
        "disk": {
            "working": disk_report(Path("/kaggle/working") if Path("/kaggle/working").exists() else OUT_DIR),
            "input": disk_report(Path("/kaggle/input") if Path("/kaggle/input").exists() else DATA_DIR),
        },
        "predictions_file": str(out_file),
        "eval_score_file": str(eval_score_file),
        "strict_score_passed": score_payload.get("passed", False),
        "acceptance": acceptance,
    }
    (OUT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_failure("strict_eval", exc)
        raise
