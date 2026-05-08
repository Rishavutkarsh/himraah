# HimRaah Submission Skeleton

## Problem

Pilgrims, families, elderly travelers, guides, porters, and first-time mountain
visitors move through remote pilgrimage corridors where connectivity is commonly
reported as unreliable after leaving Gangotri. When the internet drops, they
still need calm route-aware help for planning, language, local context, and
safety escalation.

## Users

- Pilgrims on mountain yatras
- Elderly travelers and families
- First-time visitors
- Non-Hindi or non-local-language speakers
- Local guides, porters, and remote mountain communities

## Resilience Angle

HimRaah treats offline access as a public-interest resilience problem, not a
luxury trekking feature. It is designed for low-connectivity conditions where
live search, maps, or chat services may not be available.

## Offline Route-Pack Architecture

- Bundled Gangotri -> Chirbasa -> Bhojbasa -> Gomukh route pack
- Local route facts, safety facts, phrasebook, and field-guide context
- Conservative runtime prompt/context injection
- Explicit refusal for live/current status, prices, weather, rescue, permits,
  accommodation, transport, or carrier guarantees
- Graceful uncertainty when the route pack does not contain enough information

## Gemma Role

Gemma powers a local companion that can answer messy user questions with route
context, risk level, immediate next steps, what not to do, escalation signs,
missing information, and Hindi/Hinglish support.

## Model Strategy

Primary comparison:

- Base Gemma zero-shot
- Base Gemma with HimRaah route context
- Fine-tuned HimRaah LoRA with the same route context

If LoRA improves safely, demo the tuned model. If not, use the base-with-context
offline route-pack fallback and report the SFT experiment transparently.

Start this skeleton after the first Kaggle attempt, successful or failed. Kaggle
runtime failures, dependency constraints, and fallback behavior are part of the
engineering story.

## MVP Companion Modes

- Safety check for altitude, cold, darkness, rivers/glaciers, and split groups
- Route pacing/acclimatization and resource reminders
- Local culture/ecology learning
- Hindi/Hinglish local-help phrases

The demo must include one concrete Gangotri -> Chirbasa -> Bhojbasa -> Gomukh
route scenario with safety escalation, not only generic mountain advice.

## Limitations

HimRaah does not replace doctors, certified guides, rescue teams, local
authorities, official weather, or current permit/route advisories. It must not
claim current live conditions from offline context.

## Future Extension

Vision may later support cautious trail-sign, terrain, snow-condition, or photo
interpretation, but it is not part of the judged MVP unless separately built and
evaluated.

## Risk Register

- Kaggle runtime or GPU failure:
- Dependency or wheelhouse mismatch: preflight v2 showed `bitsandbytes` missing
  and Kaggle rejected `rishavutkarsh/himraah-transformers-wheels` as a dataset
  source; fix package availability before smoke SFT.
- Offline asset loading issue:
- Unsafe advice or missing escalation:
- Overclaiming SFT improvement:
- Generic-chatbot drift:
- Weak demo differentiation:
- Vision implied without implementation:
