"""
mythos_bridge.py — Ingest WithinUsAI/claude_mythos_distilled_25k into OpenFable format.

The Mythos distillation dataset has excellent task-difficulty signal but no narrative
domain coverage. This bridge re-annotates every example with recurrence-depth metadata
so it can serve as Stage 1 pretraining for OpenFable's RDT architecture.

What it does:
  1. Loads the dataset from HuggingFace (requires: pip install datasets)
  2. Scores each example for reasoning complexity (0–1)
  3. Assigns narrative_mode (action/dialogue/exposition) and suggested_n_loops
  4. Strips the templated "Drawing from the autonomous..." preamble
  5. Writes OpenFable-format JSONL

Usage:
  python -m open_fable.data.mythos_bridge --output data/mythos_bridge.jsonl
  python -m open_fable.data.mythos_bridge --stats
  python -m open_fable.data.mythos_bridge --max-examples 500 --output data/mythos_mini.jsonl
"""

from __future__ import annotations

import json
import re
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ── Complexity priors by category ────────────────────────────────────────────

CATEGORY_BASE_COMPLEXITY: dict[str, float] = {
    "mathematical_reasoning": 0.85,   # proof-level, formal
    "advanced_coding":        0.75,   # impl + correctness
    "cybersecurity":          0.80,   # adversarial + multi-step
    "scientific_analysis":    0.70,   # synthesis + design
    "agentic_planning":       0.92,   # longest horizon
    "general_expert_qa":      0.55,   # variable
}

# Keywords that add +0.02 each (capped at 1.0)
HARD_SIGNALS = [
    "proof", "theorem", "formal verification", "adversarial", "autonomous",
    "multi-step", "12-month", "production-ready", "lock-free", "SIMD",
    "Riemann", "Strassen", "CRISPR", "quantum", "multiplexed", "concurrent",
    "zero-trust", "exploit chain", "closed-form", "time complexity",
]

TEMPLATED_PREFIX = re.compile(r"^Drawing from the autonom\w*[^.\n]*[.\n]?\s*", re.IGNORECASE)


def score_complexity(prompt: str, category: str) -> float:
    base = CATEGORY_BASE_COMPLEXITY.get(category, 0.60)
    bonus = sum(0.02 for kw in HARD_SIGNALS if kw.lower() in prompt.lower())
    return round(min(1.0, base + bonus), 3)


def loops_for(score: float) -> int:
    if score < 0.40: return 4
    if score < 0.65: return 8
    if score < 0.80: return 16
    return 32


def mode_for(score: float) -> str:
    if score < 0.40: return "action"
    if score < 0.65: return "dialogue"
    return "exposition"


def strip_preamble(text: str) -> tuple[str, bool]:
    """Remove templated preamble. Returns (cleaned, was_templated)."""
    stripped = TEMPLATED_PREFIX.sub("", text.strip())
    was_templated = stripped != text.strip()
    return stripped.strip(), was_templated


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BridgedExample:
    id: str
    source: str
    original_category: str
    complexity_score: float
    narrative_mode: str
    suggested_n_loops: int
    fable_memory_required: bool   # True when multi-step planning/state is needed
    templated_preamble_stripped: bool
    messages: list
    timestamp: str

    @classmethod
    def from_row(cls, row: dict, idx: int) -> "BridgedExample":
        msgs     = row.get("messages", [])
        category = row.get("category", "general_expert_qa")
        row_id   = row.get("id", f"mythos-{idx:05d}")
        ts       = row.get("timestamp", "")

        prompt   = next((m["content"] for m in msgs if m["role"] == "user"),      "")
        response = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

        score         = score_complexity(prompt, category)
        n_loops       = loops_for(score)
        mode          = mode_for(score)
        clean_resp, was_preamble = strip_preamble(response)

        return cls(
            id=f"openfable-bridge-{row_id}",
            source="WithinUsAI/claude_mythos_distilled_25k",
            original_category=category,
            complexity_score=score,
            narrative_mode=mode,
            suggested_n_loops=n_loops,
            fable_memory_required=(category == "agentic_planning"),
            templated_preamble_stripped=was_preamble,
            messages=[
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": clean_resp},
            ],
            timestamp=ts,
        )


# ── Public API ────────────────────────────────────────────────────────────────

class MythosBridge:
    """Pipeline wrapper — call .run() to get bridged examples."""

    HF_DATASET = "WithinUsAI/claude_mythos_distilled_25k"

    def run(self, max_examples: Optional[int] = None) -> list[BridgedExample]:
        return process_dataset(max_examples)

    def stats(self, examples: list[BridgedExample]) -> dict:
        modes      = Counter(e.narrative_mode for e in examples)
        cats       = Counter(e.original_category for e in examples)
        templated  = sum(1 for e in examples if e.templated_preamble_stripped)
        avg_score  = sum(e.complexity_score for e in examples) / max(len(examples), 1)
        loop_dist  = Counter(e.suggested_n_loops for e in examples)
        return dict(total=len(examples), avg_complexity=round(avg_score, 3),
                    templated_stripped=templated, modes=dict(modes),
                    categories=dict(cats), loop_distribution=dict(loop_dist))


def process_dataset(max_examples: Optional[int] = None) -> list[BridgedExample]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets huggingface_hub")

    ds = load_dataset("WithinUsAI/claude_mythos_distilled_25k", split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    return [BridgedExample.from_row(row, i) for i, row in enumerate(ds)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Bridge Mythos distillation dataset → OpenFable format")
    ap.add_argument("--output", default="data/mythos_bridge.jsonl")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--stats", action="store_true", help="Print stats only, no write")
    args = ap.parse_args()

    print("Loading WithinUsAI/claude_mythos_distilled_25k …")
    examples = process_dataset(args.max_examples)
    bridge   = MythosBridge()
    s        = bridge.stats(examples)

    loop_labels = {4: "action (4 loops)", 8: "dialogue (8)", 16: "exposition (16)", 32: "deep (32)"}
    print(f"\n{'='*52}")
    print(f"  Total:              {s['total']:,}")
    print(f"  Avg complexity:     {s['avg_complexity']:.3f}")
    print(f"  Preamble stripped:  {s['templated_stripped']:,} ({100*s['templated_stripped']/s['total']:.1f}%)")
    print(f"\n  Recurrence distribution:")
    for loops, count in sorted(s['loop_distribution'].items()):
        bar = "█" * (count * 30 // s['total'])
        print(f"    {loop_labels.get(loops, loops):25s} {count:5,}  {bar}")
    print(f"\n  Categories:")
    for cat, count in sorted(s['categories'].items(), key=lambda x: -x[1]):
        print(f"    {cat:30s} {count:5,}")
    print(f"{'='*52}\n")

    if args.stats:
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    print(f"Written → {args.output}  ({len(examples):,} examples)")


if __name__ == "__main__":
    main()
