# OpenFable Datasets

Two complementary training pipelines — one bridges existing signal, one generates something new.

---

## MythosBridge

**Source:** [`WithinUsAI/claude_mythos_distilled_25k`](https://huggingface.co/datasets/WithinUsAI/claude_mythos_distilled_25k)  
**Purpose:** Stage 1 pretraining — general deep reasoning with recurrence-depth annotations  
**Script:** `open_fable/data/mythos_bridge.py`

### What it does

The Mythos distillation dataset has excellent task-difficulty signal across six categories
(mathematical reasoning, advanced coding, cybersecurity, scientific analysis, agentic planning,
general expert QA) but no narrative domain coverage and no recurrence-depth annotations.

MythosBridge annotates every example with:

| Field | Description |
|---|---|
| `complexity_score` | 0–1, derived from category priors + keyword signals |
| `suggested_n_loops` | 4 / 8 / 16 / 32 — theoretically grounded recurrence target |
| `narrative_mode` | `action` / `dialogue` / `exposition` |
| `fable_memory_required` | `True` for agentic planning tasks |
| `templated_preamble_stripped` | `True` when "Drawing from the autonomous…" preamble removed |

### Recurrence distribution (full 25k)

| Mode | Examples | Loops |
|---|---|---|
| action | ~0 | 4 |
| dialogue | ~2,500 | 8 |
| exposition | ~18,000 | 16 |
| deep | ~4,500 | 32 |

### Usage

```bash
# Requires: pip install datasets huggingface_hub
python -m open_fable.data.mythos_bridge --output data/mythos_bridge.jsonl
python -m open_fable.data.mythos_bridge --stats
python -m open_fable.data.mythos_bridge --max-examples 500
```

```python
from open_fable.data import bridge_dataset
examples = bridge_dataset(max_examples=1000)
print(examples[0].suggested_n_loops)  # 32
print(examples[0].narrative_mode)     # "exposition"
```

---

## FableForge

**Purpose:** Stage 2 fine-tuning — narrative-specific reasoning with loop-depth grounding  
**Script:** `open_fable/data/fable_forge.py`  
**Novel claim:** *Narrative difficulty has a measurable recurrence requirement.*

### Why this doesn't exist anywhere else

Every existing narrative dataset (ConStory-Bench, LongPage, StoryReasoning, GOAT-70B-Storytelling)
treats depth of reasoning as an emergent property. None annotate *how much computation* a given
example needs. They are task datasets, not recurrence-training datasets.

FableForge is the first dataset where `suggested_n_loops` has a theoretically grounded basis
derived from the structural complexity of each task — not a heuristic label, not an emergent
property, but a computable function of task structure.

### Three task types

#### 1. `character_trace`
Track a named character's state (location, emotion, companions) across N scene transitions.

```
complexity = f(n_characters, n_scenes)
loops      = 4   if n_chars=1, n_scenes=3
           = 32  if n_chars=7, n_scenes=9
```

FableMemory has concrete work: the character vector must persist loop-to-loop without drift.
CoherenceProbe monitors the tracked character's entity distribution across loops.

#### 2. `coherence_challenge`
Detect and correct a single planted narrative inconsistency. Six inconsistency types,
ordered by cognitive depth required:

| Type | Loops | Why |
|---|---|---|
| `name_drift` | 4 | Surface pattern — name mismatch |
| `location_contradiction` | 8 | Spatial reasoning |
| `object_continuity` | 8 | Object state tracking |
| `timeline_error` | 16 | Temporal ordering |
| `relationship_error` | 16 | Social graph recall |
| `trait_reversal` | 32 | Character psychology + no arc provided |

CoherenceProbe detects this directly: the entity distribution for the inconsistent character
diverges between early and late loops when a trait reversal is present.

#### 3. `narrative_completion`
Generate a story continuation that satisfies N explicit constraints simultaneously.

```
complexity = f(n_characters, n_constraints)
loops      = 4   if n_chars=2, n_constraints=2
           = 32  if n_chars=6, n_constraints=8
```

Tests whether the model can hold multiple narrative requirements in latent space across
recurrence iterations without dropping any constraint.

### Example

```jsonl
{
  "id": "fable-forge-coh-000142-a3f2b1c0",
  "task_type": "coherence_challenge",
  "genre": "fantasy",
  "inconsistency_type": "trait_reversal",
  "n_characters": 4,
  "complexity_score": 0.910,
  "suggested_n_loops": 32,
  "narrative_mode": "exposition",
  "fable_memory_required": true,
  "coherence_probe_targets": ["Lyra", "Theon", "Cael", "Mira"],
  "messages": [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "fable_forge_v1"
}
```

### Usage

```bash
# No external dependencies — stdlib only
python -m open_fable.data.fable_forge --count 5000 --output data/fable_forge.jsonl
python -m open_fable.data.fable_forge --count 100 --task-type coherence_challenge --stats
python -m open_fable.data.fable_forge --count 1000 --seed 99  # deterministic
```

```python
from open_fable.data import forge_dataset

examples = forge_dataset(count=5000, seed=42)
hard = [e for e in examples if e["suggested_n_loops"] == 32]
print(f"{len(hard)} examples require 32 loops")

# Use with NarrativeDepthController
from open_fable.depth import NarrativeDepthController
ndc = NarrativeDepthController()
for ex in examples[:5]:
    loops = ndc.get_n_loops(ex["narrative_mode"])
    assert loops >= ex["suggested_n_loops"] or loops == ndc.get_n_loops("exposition")
```

---

## Two-Stage Training Pipeline

```
Stage 1: MythosBridge (25k examples)
  └── General deep reasoning: math, code, security, science, planning
  └── All annotated with suggested_n_loops
  └── Foundation for RDT recurrence dynamics

Stage 2: FableForge (scale to taste: 10k–1M)
  └── Narrative-specific: character tracing, coherence detection, constrained completion
  └── FableMemory injection validated on character_trace tasks
  └── CoherenceProbe signal validated on coherence_challenge tasks
  └── Loop-depth grounded in task structure, not heuristics
```

### Combining the datasets

```python
from open_fable.data import bridge_dataset, forge_dataset
import json, random

stage1 = bridge_dataset()           # 25k Mythos-bridged examples
stage2 = forge_dataset(count=25000) # 25k FableForge examples

# Convert BridgedExample dataclasses to dicts
stage1_dicts = [{"id": e.id, "source": e.source, "messages": e.messages,
                 "suggested_n_loops": e.suggested_n_loops,
                 "narrative_mode": e.narrative_mode} for e in stage1]

combined = stage1_dicts + stage2
random.shuffle(combined)

with open("data/openfable_combined_50k.jsonl", "w") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")

print(f"Combined: {len(combined):,} examples")
```

---

## Citation

If you use FableForge in research:

```bibtex
@misc{openfable2026fableforge,
  title   = {{FableForge}: A Synthetic Narrative Dataset with Recurrence-Depth Annotations},
  author  = {OpenCoven},
  year    = {2026},
  url     = {https://github.com/OpenCoven/open-fable},
  note    = {Part of the OpenFable project. First dataset designed around
             recurrence depth requirements for narrative reasoning tasks.}
}
```

MythosBridge is a derivative of `WithinUsAI/claude_mythos_distilled_25k` — cite that dataset
when using bridged examples.
