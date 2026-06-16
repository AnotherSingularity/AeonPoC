<!--
TEMPLATE — HuggingFace model-card structure for Aeon.
Sections marked <TODO> are intentionally unfilled. Architecture, training, and
evaluation details are completed at release time, once the public-disclosure
level and license are decided. Do not fill these in before that decision.
-->
---
license: <TODO>            # SPDX id once chosen
library_name: transformers
pipeline_tag: text-generation
tags:
  - aeon
---

# Aeon

A small, efficient language model.

## Model details

- **Developed by:** Dylan Scott (Horizon Technologies)
- **Model type:** Causal language model
- **Parameters:** <TODO>
- **Languages:** <TODO>
- **License:** <TODO — see LICENSE>
- **Architecture summary:** <TODO — public-disclosure level TBD>

## Intended use

- **Primary use:** <TODO>
- **Out of scope:** <TODO>

## How to use

```python
from aeon import AeonForCausalLM
from transformers import AutoTokenizer

model = AeonForCausalLM.from_pretrained("<checkpoint>")
tok = AutoTokenizer.from_pretrained("<checkpoint>")
# ... generate ...
```

## Training

- **Data:** <TODO>
- **Procedure:** <TODO>
- **Compute:** <TODO>

## Evaluation

- <TODO>

## Limitations and biases

- <TODO>

## Citation

Aeon, by Dylan Scott (Horizon Technologies). See `CITATION.cff` for machine-readable
citation metadata.
