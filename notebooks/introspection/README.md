# Mechanisms of Introspective Awareness — behavioral replication at 4B scale

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frnkly/papers/blob/main/notebooks/introspection/introspection_replication.ipynb)

A budget-friendly, Colab-compatible replication of the **core behavioral finding** of:

> Macar, Yang, Wang, Wallich, Ameisen & Lindsey (2026). *Mechanisms of Introspective Awareness.*
> [arXiv:2603.21396](https://arxiv.org/abs/2603.21396) · [official code](https://github.com/safety-research/introspection-mechanisms)

## The question

The paper injects "thoughts" (concept steering vectors) into Gemma3-27B's residual stream and shows the model **detects the injection at a 38.2% mean rate while claiming detection on non-injected control trials ≈ 0% of the time** (500 concepts, L=37, α=4, GPT-4.1-mini judge). The 27B model needs ≥48 GB VRAM, so this notebook asks the honest small-compute version of the question:

**Does the introspection *behavior* — detection above a ~0% false-positive floor — survive in a 4B sibling of the paper's model, using the paper's exact prompts, vector recipe, metrics, and judge rubric?**

## Contents

| File | What |
|---|---|
| [`introspection_replication.ipynb`](introspection_replication.ipynb) | The whole experiment, heavily annotated as study material. Each step states the paper's finding (with its actual numbers) *before* the code that tests it, and every compute-driven deviation is flagged with a "Paper did X / We do Y / Why" callout. |

## Requirements

- **GPU:** free Colab T4 (16 GB) runs the default config (`gemma-3-4b-it`, fp16, `QUICK` mode, ~45–75 min). A marked switch upgrades to `gemma-2-9b-it` (4-bit) for Colab Pro.
- **Hugging Face:** accept the [Gemma license](https://huggingface.co/google/gemma-3-4b-it) and put `HF_TOKEN` in Colab Secrets.
- **Optional:** `ANTHROPIC_API_KEY` for the LLM judge (≈ $0.50–1.50 with `claude-haiku-4-5`; the paper's exact judge prompts, Tables 5–7). Without a key the notebook falls back to a regex judge.

## Scope

Replicated: the concept-injection setup (§2), detection/FPR/introspection/forced-identification metrics, and a Figure-9-style layer × strength sweep. Explained-but-not-run (they need ≥48 GB and/or released-only-for-27B transcoders): the evidence-carrier/gate circuit analysis (§5), the DPO-origin experiments on OLMo-32B checkpoints (§3.3), and the abliteration / trained-bias-vector underelicitation results (§6).

> Note: until this branch is merged to `main`, open the notebook in Colab via *File → Open notebook → GitHub* and pick the branch, or fix the badge URL's branch segment.
