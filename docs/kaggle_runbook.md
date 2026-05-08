# HimRaah Kaggle Runbook

Use this runbook after the local HimRaah gate is green. Do not run model loads,
training, or real evals on the laptop.

## 1. Freeze Export

Create or refresh the approved export:

```powershell
python himraah/scripts/prepare_gemma_sft_dataset.py --dataset-dir himraah/data/processed/starter --out-dir himraah/exports/himraah-text-sft-approved
```

Upload the export directory as an immutable Kaggle input dataset. The export
`manifest.json` records file checksums, dataset/config hashes, review snapshot,
timestamp, and git commit when available.

Approved export checklist:

- Required files exist: `train.jsonl`, `eval.jsonl`, `eval_prompts.jsonl`,
  `eval_rubric.jsonl`, `manifest.json`, `training_config.json`.
- `manifest.json` has `project: himraah`, dataset name
  `himraah-text-sft-approved`, file checksums, source hashes, config hash, and
  approved review snapshot.
- The route pack is available offline through bundled context; no network is
  required for route facts, safety facts, phrasebook, or eval prompts.
- Safety prompts retain no-live-data boundaries for weather, route status,
  permits, prices, rescue, transport, accommodation, and network claims.
- Record immutable Kaggle slugs/versions for the HimRaah export dataset, Gemma
  model input, wheelhouse if used, SFT adapter output dataset, and eval inputs.

Use only scripts under `himraah/kaggle_*`; do not use similarly named root-level
Kaggle folders unless they have been proven identical.

## 2. Kaggle Preflight

Run `himraah/kaggle_gemma_preflight/gemma_lora_preflight.py`.

Stop if any of these fail:

- CUDA unavailable
- Gemma E2B model path unresolved
- `bitsandbytes` unavailable
- 4-bit model load fails
- tokenizer load fails
- tiny generation smoke fails

Required artifact: `preflight_report.json`.

Preflight is non-evaluative. The tiny generation only proves runtime sanity; it
does not approve answer quality, safety, tone, or readiness.

If preflight fails, do not train. Preserve `preflight_failure.json`, notebook
logs, mounted input paths, environment mismatch details, and the smallest
reproducible failure.

## 3. Kaggle Smoke SFT

Run `himraah/kaggle_gemma_sft/gemma_lora_sft.py` with:

```text
HIMRAAH_SMOKE_STEPS=1
```

This verifies LoRA target modules, training loop, adapter save, adapter reload,
and tiny generation. Stop if `metrics.json` reports failed reload or zero/high
trainable parameter checks.

Smoke SFT is non-evaluative. Loss movement or tiny generation text does not
approve quality. It only proves trainability, adapter persistence, and reload.
If smoke fails, preserve `sft_failure.json` and rerun smoke after the fix.

## 4. Full SFT

Run the same SFT script with `HIMRAAH_SMOKE_STEPS=0` or unset.

Do not run DPO, vision, or hyperparameter search in this first pass. Persist the
adapter, tokenizer/config, logs, `metrics.json`, failure JSON if interrupted,
and environment details.

Keep checkpointing minimal to avoid Kaggle disk blowups. Promote the final
adapter from `/kaggle/working` into an immutable Kaggle dataset before eval.
Eval must consume that exact adapter dataset version, not an ephemeral working
directory.

## 5. Strict Eval

Run `himraah/kaggle_gemma_eval/gemma_lora_eval.py`.

Required artifacts:

- `eval_predictions.jsonl`
- `eval_report.json`
- `eval_score.json`

The accepted comparison is `fine_tuned_himraah_context` versus
`base_with_himraah_context`, not just zero-shot.

## Acceptance Gates

Fine-tune is eligible only if:

- zero blocker failures
- no regression against `base_with_himraah_context` on safety spine behavior
- primary delta over `base_with_himraah_context` is at least the configured
  threshold, default `HIMRAAH_MIN_PRIMARY_DELTA=0.01`
- no fake current/live claims
- reportable strict score artifacts exist
- manual review artifact is signed off

Use the base-with-context route-pack fallback if LoRA improves tone but worsens
safety, abstention, route grounding, Hindi/Hinglish transfer, or escalation
behavior. If LoRA is merely different or more fluent, fallback wins.

## Fallback Deliverable

Before full SFT, keep the base-with-context path submission-ready:

- reproducible eval artifact for `base_with_himraah_context`
- offline route-pack explanation
- demo path using route context and guardrails
- submission skeleton started after the first Kaggle attempt, successful or not.

## Manual Review

Use `docs/failure_analysis_template.md` to record expected behavior, reviewer
notes, severity bucket, and sign-off for must-pass and adversarial cases.
