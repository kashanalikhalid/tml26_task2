# Assignment 2 Report Draft

## Problem Framing

- Define what counts as stolen, derived, and independent under the assignment rules.
- Explain why stolen-model detection matters in the white-box setting.

## Available Information

- Describe the target checkpoint and any training metadata provided with it.
- Describe the suspect model collection and how it is stored.

## Approach

- Start with checkpoint-level similarity signals.
- Add architecture, layer-name, and tensor-statistics features if needed.
- Explain how the final confidence score is calibrated.

## Baseline Features To Consider

- Parameter key overlap
- Parameter shape overlap
- Exact tensor match rate
- Sampled tensor cosine similarity
- Relative norm difference
- Similarity restricted to backbone or classifier layers

## Evaluation Plan

- Inspect top-ranked suspects manually.
- Group suspects by architecture family if metadata is available.
- Compare score distributions for likely direct copies versus clearly unrelated models.

## Failure Modes

- Independently trained models on the same data may look deceptively similar.
- Renamed keys or checkpoint wrappers can hide true overlap.
- Fine-tuned stolen models may diverge in late layers while keeping earlier features close.

## Final Submission Notes

- Record the exact score-generation pipeline used for submission.
- Keep a short changelog of experiments and leaderboard results.
