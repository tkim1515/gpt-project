# Agent Instructions

## Scope
- This repository is a small, single-file TinyGPT demo focused on character-level Shakespeare text generation.
- The main entry point is shakespeare_gpt.py.

## Run and Train
- Train: python shakespeare_gpt.py --train
- Generate: python shakespeare_gpt.py --generate
- Default behavior: if shakespeare_gpt.pt exists, it generates; otherwise it trains and then generates.
- Resume training from an existing checkpoint: python shakespeare_gpt.py --train --resume

## Data and Artifacts
- input.txt is the training corpus; it will be downloaded automatically if missing.
- shakespeare_gpt.pt is a binary model checkpoint; do not edit it manually.

## Implementation Notes
- The model is a small Transformer (TinyGPT) implemented in shakespeare_gpt.py.
- Most configuration is via CLI flags (see parse_args in shakespeare_gpt.py).
