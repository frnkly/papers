# Predictive Concept Decoders (PCD) — small-scale replication

Replication study of [**Predictive Concept Decoders: Training Scalable End-to-End
Interpretability Assistants**](https://arxiv.org/abs/2512.15712) (Huang, Choi, Johnson,
Schwettmann, Steinhardt — Transluce, Dec 2025, arXiv:2512.15712).

The paper trains an interpretability assistant end-to-end: an **encoder** compresses a
subject model's activations into a sparse list of concepts (top-k bottleneck), and a
**decoder** (subject model + LoRA) must predict the subject's behavior from those concepts
alone. There is **no official code release**, so the architecture is re-implemented from
the paper's equations and appendix, with the relevant section/figure cited at every step.

## Contents

- [`pcd_replication.ipynb`](pcd_replication.ipynb) — a Google Colab notebook, structured
  as study material: each step states the paper's claim and expected result *first*, then
  implements and runs it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frnkly/papers/blob/main/pcd/pcd_replication.ipynb)

## What it reproduces (qualitatively, at ~1/70th the paper's scale)

| Claim | Paper ref |
|---|---|
| C1 — decoder steadily improves at predicting suffix tokens through the concept bottleneck | §3.1, Fig 3 (left) |
| C2 — concepts die without the auxiliary loss; Eq 3 keeps >90% alive | §3.2, Fig 13 |
| C3 — concept precision (auto-interp) & recall scale with data, plateau/degrade without the aux loss | §3.3, Fig 3 (mid/right) |
| Mini probe: implanted concept surfacing through the bottleneck | §5.3, Fig 11 |

Skipped for compute/data reasons (documented in the notebook's final cell): SynthSys QA
finetuning (§4), SAE/KL-SAE baselines (Fig 4), jailbreak & secret-hint case studies
(§5.1–5.2).

## Compute

- **Default:** free Colab T4 (16 GB) — subject model `Llama-3.2-1B-Instruct`
  (vs the paper's `Llama-3.1-8B-Instruct`), ~1M-token budget (vs 9M–144M),
  total ≈ 1.5–2.5 h. Presets for L4 / A100 (3B subject) included.
- **Optional:** `ANTHROPIC_API_KEY` for Claude-based auto-interp scoring (≈ $1–3 with
  Haiku); a free heuristic fallback keeps the notebook fully runnable without a key.
- `meta-llama/*` models require a Hugging Face token (license acceptance); an ungated
  fallback (`Qwen/Qwen2.5-1.5B-Instruct`) is one commented line away.

## Notes on fidelity

Faithful to the paper: Eq 1 encoder (incl. initialization W_emb = W_encᵀ, unit-norm rows),
prefix/middle/suffix pretraining objective (Eq 2, 16/16/16 tokens), soft-token patching at
ℓ_write = 0, mid-stack ℓ_read, 8× dictionary expansion, k = 16, auxiliary loss (Eq 3) with
the paper's ε_aux and scaled k_aux/window, and A.1's optimizer settings (lr 1e-4, wd 0.01,
cosine, LoRA α=32/dropout 0.05, INSTRUCT_PREFIX). Every deviation forced by compute is
marked inline with a "📄 Paper did X / 🔧 We do Y / 💡 Why" callout.

The notebook's core cells (encoder, patching, both losses, the full training loop, and the
evaluation code) were validated end-to-end with a tiny random-weight Llama before
publishing.
