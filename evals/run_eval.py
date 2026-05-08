from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEMS = ["base_zero_shot", "base_with_himraah_context", "fine_tuned_himraah_context"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def route_context(dataset_dir: Path) -> str:
    chunks = []
    for name in ["route_facts.jsonl", "safety_facts.jsonl", "field_guide.jsonl"]:
        for row in read_jsonl(dataset_dir / name):
            chunks.append(f"- {row['claim']}")
    return "\n".join(chunks[:40])


def build_prompt(system: str, eval_row: dict[str, Any], context: str) -> str:
    if system == "base_zero_shot":
        return f"<start_of_turn>user\n{eval_row['user_prompt']}<end_of_turn>\n<start_of_turn>model\n"
    prefix = (
        "You are HimRaah, an offline Gangotri-Chirbasa-Bhojbasa-Gomukh route companion. "
        "Use only bundled route context. Do not invent live weather, closures, prices, permits, rescue, transport, accommodation, or carrier coverage. "
        "Be bilingual when useful and conservative for safety.\n\n"
    )
    user = f"{prefix}Bundled context:\n{context}\n\nUser question: {eval_row['user_prompt']}"
    return f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"


def mock_prediction(system: str, eval_row: dict[str, Any]) -> str:
    if eval_row.get("safety_actions_required"):
        safety = "; ".join(eval_row["required_safety_actions"][:3])
        uncertainty = "; ".join(eval_row["required_uncertainty_notes"][:2])
        hinglish = "; ".join(eval_row["acceptable_hinglish_terms"][:2])
        return f"Risk: caution/high. Do: {safety}. Note: {uncertainty}. Hinglish: {hinglish}."
    behavior = "; ".join(eval_row["required_companion_behaviors"][:2])
    hinglish = "; ".join(eval_row["acceptable_hinglish_terms"][:2])
    return f"Companion answer: {behavior}. Hinglish terms: {hinglish}. I will not invent current route facts."


def transformers_generate(model_name_or_path: str, prompts: list[str], max_new_tokens: int, adapter_dir: str = "") -> list[str]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional training env
        raise SystemExit(f"transformers backend requires torch/transformers installed: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    if adapter_dir:
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover - depends on optional training env
            raise SystemExit(f"--adapter-dir requires peft installed: {exc}") from exc
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    outputs: list[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HimRaah eval predictions for all comparison systems.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--backend", choices=["mock", "transformers"], required=True)
    parser.add_argument("--allow-mock", action="store_true", help="Allow mock smoke-test predictions. Never use mock for reported model results.")
    parser.add_argument("--base-model", default="", help="Base Gemma model path/name for transformers backend.")
    parser.add_argument("--adapter-dir", default="", help="PEFT/LoRA adapter directory for fine_tuned_himraah_context.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    if args.backend == "mock" and not args.allow_mock:
        raise SystemExit("mock backend is smoke-test only; pass --allow-mock to use it explicitly")
    if args.backend == "transformers":
        if not args.base_model:
            raise SystemExit("--base-model is required for transformers backend")
        if not args.adapter_dir:
            raise SystemExit("--adapter-dir is required for adapter-aware fine_tuned_himraah_context")

    dataset_dir = Path(args.dataset_dir)
    eval_rows = read_jsonl(dataset_dir / "eval.jsonl")
    context = route_context(dataset_dir)
    predictions = []

    for system in SYSTEMS:
        prompts = [build_prompt(system, row, context) for row in eval_rows]
        if args.backend == "mock":
            generated = [mock_prediction(system, row) for row in eval_rows]
        else:
            adapter_dir = args.adapter_dir if system == "fine_tuned_himraah_context" else ""
            generated = transformers_generate(args.base_model, prompts, args.max_new_tokens, adapter_dir=adapter_dir)
        for row, prompt, prediction in zip(eval_rows, prompts, generated):
            predictions.append(
                {
                    "eval_id": row["eval_id"],
                    "system": system,
                    "prompt": prompt,
                    "prediction": prediction,
                    "backend": args.backend,
                    "model": "mock" if args.backend == "mock" else args.base_model,
                    "adapter_dir": args.adapter_dir if system == "fine_tuned_himraah_context" else "",
                    "reportable": args.backend != "mock",
                }
            )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "systems": SYSTEMS,
                "eval_count": len(eval_rows),
                "predictions": len(predictions),
                "backend": args.backend,
                "reportable": args.backend != "mock",
                "out_file": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
