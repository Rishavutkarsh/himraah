# HimRaah Preflight Runs

## 2026-05-08 Kaggle Preflight Attempts

Repo: `Rishavutkarsh/himraah`

## Local Gate

- HimRaah tests: passed, 25 tests.
- Dataset validation: passed with known current-status warning for permit/rule facts.
- Export dataset: `rishavutkarsh/himraah-text-sft-approved`
- Latest export commit: `3613a0f16fd6494aa6ca16f05f9a7c8b5b8a369b`
- Export bundle sha256: `c8fe2524b59e5f29160d5a030490cdedd0faf76dc28c534f1c2f8ccfcc77fa9b`

## Kernel Version 1

- Kernel: `rishavutkarsh/himraah-gemma-e2b-preflight`
- Result: failed.
- Taxonomy: `dataset_manifest`
- Evidence: `preflight_failure.json`
- Cause: Kaggle mounted inputs under `/kaggle/input/datasets` and `/kaggle/input/models`, while the script expected `/kaggle/input/himraah-text-sft-approved`.
- Action taken: updated HimRaah Kaggle scripts to resolve the approved manifest dynamically under `/kaggle/input`.

## Kernel Version 2

- Kernel: `rishavutkarsh/himraah-gemma-e2b-preflight`
- Result: failed.
- Taxonomy: `package_or_wheel`
- Evidence: `preflight_failure.json`
- Cause: `bitsandbytes` is not installed in the Kaggle runtime.
- Additional signal: Kaggle rejected `rishavutkarsh/himraah-transformers-wheels` as an invalid dataset source during kernel push, so the intended offline wheelhouse is not currently mounted.
- GPU evidence: Tesla P100 available; `nvidia-smi.ok` was true.
- Dataset evidence: approved HimRaah dataset manifest was validated before package failure.

## Decision

Stop before smoke SFT or full SFT. Next work must fix package/wheel availability only:

- Create or repair a valid private Kaggle wheelhouse dataset containing `bitsandbytes` and any required offline dependencies, or
- Use a Kaggle image/input strategy where `bitsandbytes` is already available, while keeping internet disabled.

Do not change prompts, route data, labels, SFT examples, tokenizer settings, or training config based on these preflight failures.
