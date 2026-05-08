"""Kaggle GPU Gemma E2B LoRA SFT for HimRaah text-only route companion."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


DATASET_NAME = "himraah-text-sft-approved"
DATA_DIR = Path(os.environ.get("HIMRAAH_DATA_DIR", "/kaggle/input/himraah-text-sft-approved"))
OUT_DIR = Path(os.environ.get("HIMRAAH_OUT_DIR", "/kaggle/working/himraah_gemma_e2b_lora_sft"))
SEED = int(os.environ.get("HIMRAAH_SEED", "17"))
SMOKE_STEPS = int(os.environ.get("HIMRAAH_SMOKE_STEPS", "0"))
MAX_LORA_TRAINABLE_PCT = float(os.environ.get("HIMRAAH_MAX_LORA_TRAINABLE_PCT", "10.0"))
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


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assert_himraah_data_dir() -> tuple[Path, dict]:
    data_dir = resolve_himraah_data_dir()
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise RuntimeError(f"missing HimRaah dataset files under {data_dir}: {missing}")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("project") != "himraah" or manifest.get("dataset_name") != DATASET_NAME:
        raise RuntimeError(f"wrong dataset manifest for HimRaah: {manifest}")
    return data_dir, manifest


def resolve_himraah_data_dir() -> Path:
    if all((DATA_DIR / name).exists() for name in REQUIRED_FILES):
        return DATA_DIR
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for manifest_path in input_root.rglob("manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if manifest.get("project") == "himraah" and manifest.get("dataset_name") == DATASET_NAME:
                candidate = manifest_path.parent
                if all((candidate / name).exists() for name in REQUIRED_FILES):
                    return candidate
    return DATA_DIR


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


def load_config(path: Path) -> dict:
    config = {
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
    }
    config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


class CompletionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, completion_only_loss: bool):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.completion_only_loss = completion_only_loss

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        full = self.tokenizer(row["text"], max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            prompt = self.tokenizer(row["prompt"], max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")["input_ids"][0]
            labels[: min(prompt.numel(), labels.numel())] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def package_versions() -> dict:
    versions = {}
    for name in ["torch", "transformers", "peft", "accelerate", "tokenizers", "bitsandbytes"]:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable:{exc}"
    return versions


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


def mounted_inputs() -> list[str]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    return sorted(str(path) for path in root.iterdir())


def failure_taxonomy(exc: BaseException) -> str:
    text = str(exc).lower()
    if "cuda" in text or "gpu" in text or "memory" in text:
        return "gpu_or_memory"
    if "manifest" in text or "dataset" in text:
        return "dataset_manifest"
    if "lora" in text or "trainable" in text or "adapter" in text:
        return "lora_or_adapter"
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
        "run_mode": "smoke_sft" if SMOKE_STEPS else "full_sft",
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
    (OUT_DIR / "sft_failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parameter_report(model) -> dict:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_pct = round((trainable_params / total_params) * 100, 6) if total_params else 0.0
    if trainable_params <= 0:
        raise RuntimeError("LoRA configuration produced zero trainable parameters.")
    if trainable_pct > MAX_LORA_TRAINABLE_PCT:
        raise RuntimeError(f"LoRA trainable percentage is implausibly high: {trainable_pct}%. Refusing likely full-model training.")
    return {"total_params": total_params, "trainable_params": trainable_params, "trainable_pct": trainable_pct}


def quantization_config_report() -> dict:
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    }


def reload_adapter_smoke(model_path: Path, adapter_dir: Path, tokenizer) -> dict:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    prompt = "<start_of_turn>user\nSay HimRaah adapter reload OK.<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    new_tokens = int(generated.shape[-1] - inputs["input_ids"].shape[-1])
    del model
    del base
    torch.cuda.empty_cache()
    return {"adapter_reload_ok": True, "tiny_generate_new_tokens": new_tokens}


def main() -> None:
    global LAST_CHECKPOINT
    wheel_status = install_transformers_wheel()
    LAST_CHECKPOINT = "wheel_install_checked"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Do not run HimRaah Gemma E2B SFT on CPU.")
    torch.cuda.reset_peak_memory_stats(0)
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir, manifest = assert_himraah_data_dir()
    LAST_CHECKPOINT = "dataset_manifest_validated"
    model_path = find_model_path()
    LAST_CHECKPOINT = "model_path_resolved"
    train_rows = read_jsonl(data_dir / "train.jsonl")
    eval_rows = read_jsonl(data_dir / "eval.jsonl")
    config = load_config(data_dir / "training_config.json")
    LAST_CHECKPOINT = "data_and_config_loaded"

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import bitsandbytes  # noqa: F401
    LAST_CHECKPOINT = "packages_imported"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    LAST_CHECKPOINT = "model_loaded_4bit"
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = ["linear"] if getattr(model.config, "model_type", "") == "gemma4" else list(config["target_modules"])
    lora_r = min(int(config["lora_r"]), 2) if target_modules == ["linear"] else int(config["lora_r"])
    lora_config = LoraConfig(r=lora_r, lora_alpha=int(config["lora_alpha"]), lora_dropout=float(config["lora_dropout"]), bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    params = parameter_report(model)
    LAST_CHECKPOINT = "lora_wrapped_and_params_checked"

    train_ds = CompletionDataset(train_rows, tokenizer, int(config["max_length"]), bool(config["completion_only_loss"]))
    eval_ds = CompletionDataset(eval_rows, tokenizer, int(config["max_length"]), bool(config["completion_only_loss"]))
    train_loader = DataLoader(train_ds, batch_size=int(config["per_device_train_batch_size"]), shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    grad_accum = int(config["gradient_accumulation_steps"])
    target_batches = max(1, math.ceil(len(train_loader) * float(config["num_train_epochs"])))
    if SMOKE_STEPS:
        target_batches = min(target_batches, max(1, SMOKE_STEPS * grad_accum))
    total_steps = max(1, math.ceil(target_batches / grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    global_step = 0
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= target_batches:
            break
        batch = {key: value.to(model.device) for key, value in batch.items()}
        loss = model(**batch).loss / grad_accum
        loss.backward()
        losses.append(float(loss.item() * grad_accum))
        if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == target_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print({"step": global_step, "train_loss": losses[-1]})
            LAST_CHECKPOINT = f"optimizer_step_{global_step}"

    model.eval()
    eval_losses: list[float] = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            eval_losses.append(float(model(**batch).loss.item()))

    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    LAST_CHECKPOINT = "adapter_saved"
    if SMOKE_STEPS:
        del model
        torch.cuda.empty_cache()
    reload_smoke = reload_adapter_smoke(model_path, adapter_dir, tokenizer) if SMOKE_STEPS else {"adapter_reload_ok": "not_run_full_sft"}
    LAST_CHECKPOINT = "adapter_reload_smoke_complete" if SMOKE_STEPS else "full_sft_complete"
    metrics = {
        "project": "himraah",
        "stage": "sft",
        "ok": True,
        "quality_inference": "none from smoke SFT; full SFT requires strict eval before model-quality claims",
        "dataset_name": DATASET_NAME,
        "run_mode": "smoke_sft" if SMOKE_STEPS else "full_sft",
        "smoke_steps_requested": SMOKE_STEPS,
        "manifest": manifest,
        "config": config,
        "model_path": str(model_path),
        "resolved_paths": {
            "data_dir": str(data_dir.resolve()),
            "out_dir": str(OUT_DIR.resolve()),
            "model_path": str(model_path.resolve()),
            "adapter_dir": str(adapter_dir.resolve()),
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "cuda_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 3),
        "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
        "quantized_4bit": True,
        "quantization_config": quantization_config_report(),
        "lora_target_modules": target_modules,
        "lora_r_effective": lora_r,
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "trainable_pct": params["trainable_pct"],
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "transformers_wheel": wheel_status,
        "package_versions": package_versions(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "nvidia_smi": command_output(["nvidia-smi"]),
        "disk": {
            "working": disk_report(Path("/kaggle/working") if Path("/kaggle/working").exists() else OUT_DIR),
            "input": disk_report(Path("/kaggle/input") if Path("/kaggle/input").exists() else data_dir),
        },
        "adapter_reload_smoke": reload_smoke,
        "train_steps": global_step,
        "target_batches": target_batches,
        "gradient_accumulation_steps": grad_accum,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_mean": sum(losses) / max(1, len(losses)),
        "eval_loss": sum(eval_losses) / max(1, len(eval_losses)),
    }
    metrics["eval_ppl"] = math.exp(min(float(metrics["eval_loss"]), 20))
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "sample_eval_rows.json").write_text(json.dumps(eval_rows[:5], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote HimRaah Gemma E2B LoRA adapter and metrics to {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_failure("sft", exc)
        raise
