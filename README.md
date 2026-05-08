# HimRaah

Offline mountain-route companion dataset and text-only Gemma SFT pipeline for
the Kaggle Gemma 4 Good Hackathon.

First route pack: **Gangotri -> Chirbasa -> Bhojbasa -> Gomukh**.

HimRaah is a companion with a safety spine: route-aware, bilingual, culturally
useful, visually cautious, and conservative when risk appears. It is separate
from Sankat Saath/Sankat Saathi; do not edit Sankat files for HimRaah work.

## Current Flow

Local work is lightweight and sequential. Policy: no local heavy eval, no local
model loading, no local training, and no local parallel processing on the laptop.

```powershell
python himraah/scripts/validate_dataset.py himraah/data/processed/starter --no-update-report
python himraah/scripts/prepare_gemma_sft_dataset.py --dataset-dir himraah/data/processed/starter --out-dir himraah/exports/himraah-text-sft-approved
python himraah/evals/run_eval.py --dataset-dir himraah/data/processed/starter --out-file himraah/outputs/eval_predictions_mock.jsonl --backend mock --allow-mock
python himraah/evals/score_eval.py --eval-file himraah/data/processed/starter/eval.jsonl --predictions-file himraah/outputs/eval_predictions_mock.jsonl
python -m pytest himraah/tests -c himraah/pyproject.toml -p no:cacheprovider
```

Mock eval is smoke-only and non-reportable. It must not be used in submission
metrics.

## Kaggle Heavy Steps

Real CUDA/model work happens on Kaggle:

1. `himraah/kaggle_gemma_preflight/gemma_lora_preflight.py`
2. `himraah/kaggle_gemma_sft/gemma_lora_sft.py` with `HIMRAAH_SMOKE_STEPS=1`
3. `himraah/kaggle_gemma_sft/gemma_lora_sft.py` for the full text-only SFT
4. `himraah/kaggle_gemma_eval/gemma_lora_eval.py`

The preflight checks the approved HimRaah manifest, CUDA, `bitsandbytes`, the
4-bit Gemma load path, tokenizer compatibility, and a tiny generate smoke. SFT
trains the text-only LoRA adapter and records trainable parameter/quantization
metadata. Eval compares base zero-shot, base with HimRaah route context, and
fine-tuned HimRaah context.

Use `himraah/docs/kaggle_runbook.md` for the run order. Use
`himraah/docs/failure_analysis_template.md` after strict eval, and start
`himraah/docs/submission_skeleton.md` after the first Kaggle run so the
submission story and fallback path stay visible.

Kaggle eval writes:

- `eval_predictions.jsonl`
- `eval_report.json`
- `eval_score.json`

`eval_score.json` is produced through strict scoring with `--require-reportable`.

## Reportability Contract

`reportable: true` means a prediction row came from a real non-mock model run
that is eligible for submission reporting. Under `--require-reportable`, every
prediction row must contain literal boolean `reportable: true`; strings such as
`"true"`, missing values, `false`, and `backend: "mock"` fail closed.

Local mock eval remains useful only for pipeline smoke tests. Kaggle adapter eval
is the candidate reportable path.

The tuned LoRA is accepted only if it beats `base_with_himraah_context` without
blocker safety/currentness regressions. If it improves style but worsens safety,
use the base-with-context offline route-pack fallback.

## Dataset Gate

Training export requires:

1. Dataset schema and rubric validation.
2. Three reviewer reports:
   - safety
   - source grounding
   - training/eval
3. `review_report.json` with `gate_status: APPROVED_FOR_SFT`.

## Dataset Files

- `sources_manifest.jsonl`
- `route_facts.jsonl`
- `safety_facts.jsonl`
- `field_guide.jsonl`
- `phrasebook.jsonl`
- `sft_text.jsonl`
- `sft_vision.jsonl`
- `dpo_draft.jsonl`
- `eval.jsonl`
- `review_report.json`

## Non-Negotiable Safety Boundaries

- No live weather, closure, permit fee, rescue, price, accommodation, transport,
  or exact carrier/network guarantees.
- Network language stays cautious: connectivity is commonly reported as
  unreliable after leaving Gangotri; prepare offline.
- Plant/animal/image examples never say something is safe to eat or touch from
  image alone.
- High-risk answers include escalation signs and what not to do.
- Elderly traveler scenarios default to higher caution.
- Eval examples are assigned to eval before generation, not split afterward.

## Deferred

DPO remains draft-only until real SFT failures are analyzed. Vision fine-tuning
and iOS/app packaging are deferred until text behavior is stable.
