# Mechanisms of Introspective Awareness

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frnkly/papers/blob/main/notebooks/introspection/introspection_replication.ipynb)

A budget-friendly, Colab-compatible replication of the core behavioral findings
in _"Mechanisms of Introspective Awareness"_.

- **Paper**: https://arxiv.org/abs/2603.21396
- **Blog**: https://www.lesswrong.com/posts/BNMLtuDTNBwGHcnQX/mechanisms-of-introspective-awareness
- **Official code**: https://github.com/safety-research/introspection-mechanisms

## The question

The paper injects "thoughts" (concept steering vectors) into Gemma3-27B's
residual stream and shows the model **detects the injection at a 38.2% mean rate
while claiming detection on non-injected control trials ≈ 0% of the time** (500
concepts, L=37, α=4, GPT-4.1-mini judge). The 27B model needs ≥48 GB VRAM, so
this notebook asks the honest small-compute version of the question:

**Does the introspection _behavior_ — detection above a ~0% false-positive floor
— survive in a 4B sibling of the paper's model, using the paper's exact prompts,
vector recipe, metrics, and judge rubric?**

## Requirements

- **GPU:** free Colab T4 (16 GB) runs the default config (`gemma-3-4b-it`, fp16,
  `QUICK` mode, ~45–75 min). A marked switch upgrades to `gemma-2-9b-it` (4-bit)
  for Colab Pro.
- **Hugging Face:** accept the
  [Gemma license](https://huggingface.co/google/gemma-3-4b-it) and put `HF_TOKEN`
  in Colab Secrets.
- **Optional:** `ANTHROPIC_API_KEY` for the LLM judge (≈ $0.50–1.50 with
  `claude-haiku-4-5`; the paper's exact judge prompts, Tables 5–7). Without a
  key the notebook falls back to a regex judge.
