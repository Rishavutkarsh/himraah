"""Kaggle preflight for HimRaah Gemma E2B text LoRA SFT."""

from __future__ import annotations

import importlib
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
OUT_DIR = Path(os.environ.get("HIMRAAH_OUT_DIR", "/kaggle/working/himraah_gemma_e2b_preflight"))
REQUIRED_FILES = ["train.jsonl", "eval.jsonl", "eval_prompts.jsonl", "eval_rubric.jsonl", "manifest.json", "training_config.json"]
LAST_CHECKPOINT = "start"


def install_transformers_wheel() -> str:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return "input_root_missing"
    transformer_wheels = list(input_root.rglob("transformers-5.8.0-py3-none-any.whl"))
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


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def assert_himraah_data_dir() -> tuple[Path, dict]:
    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"missing HimRaah dataset files under {DATA_DIR}: {missing}")
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("project") != "himraah" or manifest.get("dataset_name") != DATASET_NAME:
        raise RuntimeError(f"wrong dataset manifest for HimRaah: {manifest}")
    return DATA_DIR, manifest


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
    raise RuntimeError("No mounted Gemma E2B-IT Transformers model path found.")


def require_package(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


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


def failure_taxonomy(exc: BaseException) -> str:
    text = str(exc).lower()
    if "cuda" in text or "gpu" in text or "memory" in text:
        return "gpu_or_memory"
    if "manifest" in text or "dataset" in text or "missing himraah dataset" in text:
        return "dataset_manifest"
    if "bitsandbytes" in text or "transformers" in text or "package" in text or "wheel" in text:
        return "package_or_wheel"
    if "model path" in text or "gemma" in text or "mounted" in text:
        return "model_mount"
    return "script_bug_or_unknown"


def mounted_inputs() -> list[str]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    return sorted(str(path) for path in root.iterdir())


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
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python": {"executable": sys.executable, "version": sys.version, "platform": platform.platform()},
    }
    (OUT_DIR / "preflight_failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def quantization_config_dict() -> dict:
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    }


def has_4bit_layers(model) -> bool:
    return any("4bit" in module.__class__.__name__.lower() for module in model.modules())


def main() -> None:
    global LAST_CHECKPOINT
    wheel_status = install_transformers_wheel()
    LAST_CHECKPOINT = "wheel_install_checked"
    data_dir, manifest = assert_himraah_data_dir()
    LAST_CHECKPOINT = "dataset_manifest_validated"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    versions = {name: require_package(name) for name in ["torch", "transformers", "peft", "accelerate", "tokenizers", "bitsandbytes"]}
    LAST_CHECKPOINT = "packages_imported"

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HimRaah Gemma E2B SFT preflight.")
    model_path = find_model_path()
    LAST_CHECKPOINT = "model_path_resolved"
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    LAST_CHECKPOINT = "config_tokenizer_loaded"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats(0)
    memory_before = torch.cuda.memory_allocated(0)
    requested_quantization = quantization_config_dict()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
    LAST_CHECKPOINT = "model_loaded_4bit"
    prompt = "<start_of_turn>user\nSay HimRaah preflight OK.<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    LAST_CHECKPOINT = "tiny_generation_complete"
    smoke_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
    observed_4bit_layers = has_4bit_layers(model)
    memory_after = torch.cuda.memory_allocated(0)
    memory_peak = torch.cuda.max_memory_allocated(0)
    gpu_props = torch.cuda.get_device_properties(0)
    report = {
        "project": "himraah",
        "stage": "preflight",
        "ok": True,
        "quality_inference": "none; tiny generation is runtime sanity only",
        "dataset_name": DATASET_NAME,
        "data_dir": str(data_dir),
        "manifest": manifest,
        "train_rows": count_jsonl(data_dir / "train.jsonl"),
        "eval_rows": count_jsonl(data_dir / "eval.jsonl"),
        "eval_prompt_rows": count_jsonl(data_dir / "eval_prompts.jsonl"),
        "model_path": str(model_path),
        "resolved_paths": {
            "data_dir": str(data_dir.resolve()),
            "out_dir": str(OUT_DIR.resolve()),
            "model_path": str(model_path.resolve()),
        },
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
        "cuda_version": torch.version.cuda,
        "cuda_memory_before_gb": round(memory_before / (1024**3), 3),
        "cuda_memory_after_4bit_load_gb": round(memory_after / (1024**3), 3),
        "cuda_max_memory_allocated_gb": round(memory_peak / (1024**3), 3),
        "tiny_generate_new_tokens": smoke_tokens,
        "requested_quantization": requested_quantization,
        "observed_4bit_layers": observed_4bit_layers,
        "package_versions": versions,
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
        "transformers_wheel": wheel_status,
        "load_settings": {
            "torch_dtype": "float16",
            "device_map": {"": 0},
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        },
    }
    del model
    torch.cuda.empty_cache()
    (OUT_DIR / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_failure("preflight", exc)
        raise
