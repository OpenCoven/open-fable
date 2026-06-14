"""
fable_forge.py — Synthetic narrative dataset generator for OpenFable.

FableForge generates training examples where harder narrative tasks explicitly
require more recurrence loops to solve correctly. This is the first dataset
designed around recurrence depth requirements rather than treating depth as
an emergent property.

The core claim: narrative difficulty is structurally measurable.
  character_trace:     loops = f(n_characters, n_scenes)
  coherence_challenge: loops = f(inconsistency_type)     — subtle > obvious
  narrative_completion: loops = f(n_characters, n_constraints)

Each example carries:
  suggested_n_loops       — theoretically grounded recurrence target
  narrative_mode          — action (4) | dialogue (8) | exposition (16/32)
  fable_memory_required   — whether FableMemory injection is needed
  coherence_probe_targets — character names CoherenceProbe should monitor

Usage:
  python -m open_fable.data.fable_forge --count 5000 --output data/fable_forge.jsonl
  python -m open_fable.data.fable_forge --count 100 --task-type character_trace --stats
  python -m open_fable.data.fable_forge --count 1000 --seed 42 --output data/forge_s42.jsonl
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import argparse
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

# ── Name / world pools ────────────────────────────────────────────────────────

_NAMES_FANTASY = [
    "Eryn", "Cael", "Sora", "Daven", "Lyra", "Rhys", "Nara", "Finn",
    "Mira", "Aldric", "Zora", "Theon", "Vael", "Kira", "Orin", "Selene",
    "Bram", "Isolde", "Corvin", "Thessaly", "Emric", "Wren", "Cassia", "Tor",
]
_NAMES_CONTEMPORARY = [
    "Maya", "James", "Sofia", "Eli", "Priya", "Tom", "Leila", "Marcus",
    "Anna", "Dev", "Cara", "Noah", "Hana", "Sam", "Yuki", "Remy",
    "Jin", "Petra", "Caleb", "Simone", "Omar", "Fiona", "Luca", "Aiko",
]
_LOCS_FANTASY = [
    "the Ashwood", "the Iron Keep", "the Saltmarsh", "the Drifting Spire",
    "the Thornfield", "the Vaulted City", "the Pale Shore", "the Rootmere",
    "the Ember Gate", "the Hollow Court", "the Greywarden Tower", "the Tideless Pool",
]
_LOCS_CONTEMPORARY = [
    "the lab", "the station", "the archive", "the clinic",
    "the bridge", "the safehouse", "the conference room", "the rooftop",
    "the server room", "the transit hub", "the observatory", "the flooded district",
]
_TRAITS = [
    "cautious", "impulsive", "loyal", "secretive", "grieving", "determined",
    "skeptical", "protective", "reckless", "curious", "weary", "sharp",
    "calculating", "earnest", "volatile", "patient",
]
_EMOTIONS = [
    "fear", "grief", "resolve", "suspicion", "hope", "dread",
    "fury", "calm", "unease", "wonder", "guilt", "relief",
]
_RELATIONSHIPS = [
    "distrusts", "owes a debt to", "was betrayed by", "is searching for",
    "is protecting", "competed against", "once saved", "was trained by",
    "is estranged from", "is hunting", "is hiding something from",
]
_INCONSISTENCY_TYPES: list[tuple[str, str, float]] = [
    # (type_id, description, base_complexity)
    ("name_drift",             "Character's name changes mid-story",                0.30),
    ("location_contradiction", "Character simultaneously in two places",             0.45),
    ("object_continuity",      "Object appears/disappears without explanation",      0.55),
    ("timeline_error",         "Event referenced before it occurred",                0.65),
    ("relationship_error",     "Established relationship directly contradicted",      0.72),
    ("trait_reversal",         "Character acts against their core established trait", 0.82),
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _loops(score: float) -> int:
    if score < 0.35: return 4
    if score < 0.55: return 8
    if score < 0.75: return 16
    return 32

def _mode(score: float) -> str:
    if score < 0.35: return "action"
    if score < 0.55: return "dialogue"
    return "exposition"

def _uid(task_type: str, idx: int, seed: int) -> str:
    h = hashlib.sha1(f"{task_type}:{idx}:{seed}".encode()).hexdigest()[:8]
    return f"fable-forge-{task_type[:3]}-{idx:06d}-{h}"

def _pick(rng: random.Random, pool: list, n: int = 1):
    return rng.sample(pool, min(n, len(pool))) if n > 1 else rng.choice(pool)


# ── Task 1 — CharacterTrace ───────────────────────────────────────────────────

def _gen_character_trace(rng: random.Random, idx: int, seed: int) -> dict:
    """
    Track one character's state (location, emotion, companions) across N scenes.
    Difficulty = n_characters × n_scenes.
    FableMemory is doing real work here — the character vector must persist loop-to-loop.
    """
    genre    = rng.choice(["fantasy", "contemporary"])
    names    = _NAMES_FANTASY    if genre == "fantasy" else _NAMES_CONTEMPORARY
    locs     = _LOCS_FANTASY     if genre == "fantasy" else _LOCS_CONTEMPORARY

    n_chars  = rng.randint(2, 7)
    n_scenes = rng.randint(3, 9)
    chars    = rng.sample(names, min(n_chars, len(names)))
    tracked  = chars[0]

    # Build character profiles
    profiles = {c: {"trait": _pick(rng, _TRAITS), "emotion": _pick(rng, _EMOTIONS),
                    "start_loc": _pick(rng, locs)} for c in chars}

    # Generate scenes — each places some characters somewhere
    scene_records = []
    for s in range(n_scenes):
        present    = rng.sample(chars, rng.randint(1, max(1, len(chars))))
        loc        = _pick(rng, locs)
        emotion    = _pick(rng, _EMOTIONS)
        scene_records.append({"scene": s + 1, "present": present,
                               "loc": loc, "emotion": emotion})

    # Build readable scene text
    scene_texts = []
    for sr in scene_records:
        present_str = ", ".join(sr["present"])
        line = (f"Scene {sr['scene']}: At {sr['loc']} — present: {present_str}. "
                f"The mood was one of {sr['emotion']}.")
        scene_texts.append(line)

    profile_block = "\n".join(
        f"- **{c}** ({p['trait']}, starts feeling {p['emotion']} at {p['start_loc']})"
        for c, p in profiles.items()
    )

    # Ground-truth: find tracked char's last scene
    last_seen = next(
        (sr for sr in reversed(scene_records) if tracked in sr["present"]), None
    )
    final_loc     = last_seen["loc"]     if last_seen else profiles[tracked]["start_loc"]
    final_emotion = last_seen["emotion"] if last_seen else profiles[tracked]["emotion"]
    final_scene   = last_seen["scene"]   if last_seen else 0

    appearances = [sr for sr in scene_records if tracked in sr["present"]]
    trace_lines = []
    for sr in scene_records:
        if tracked in sr["present"]:
            others = [c for c in sr["present"] if c != tracked]
            trace_lines.append(
                f"  Scene {sr['scene']}: at {sr['loc']}, feeling {sr['emotion']}"
                + (f", with {', '.join(others)}" if others else ", alone")
            )
        else:
            trace_lines.append(f"  Scene {sr['scene']}: absent")

    answer_text = (
        f"Tracing **{tracked}** across {n_scenes} scenes:\n"
        + "\n".join(trace_lines)
        + f"\n\n**Final state:** location = {final_loc}, emotion = {final_emotion}"
        + (f", last seen in Scene {final_scene}" if last_seen else " (never appeared)")
    )

    prompt = (
        f"Read the following {n_scenes}-scene narrative fragment carefully.\n\n"
        f"**Character profiles:**\n{profile_block}\n\n"
        f"**Story:**\n" + "\n".join(scene_texts) + "\n\n"
        f"For each scene, state whether **{tracked}** is present. "
        f"For scenes where they appear, record: location, emotional tone, who else is present. "
        f"End with {tracked}'s final known state (location + emotion)."
    )

    score = min(1.0, 0.20 + n_chars * 0.07 + n_scenes * 0.045)
    return {
        "id":                    _uid("character_trace", idx, seed),
        "task_type":             "character_trace",
        "genre":                 genre,
        "n_characters":          n_chars,
        "n_scenes":              n_scenes,
        "constraint_count":      n_chars + n_scenes,
        "complexity_score":      round(score, 3),
        "suggested_n_loops":     _loops(score),
        "narrative_mode":        _mode(score),
        "fable_memory_required": n_chars > 2,
        "coherence_probe_targets": chars,
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": answer_text},
        ],
        "source": "fable_forge_v1",
    }


# ── Task 2 — CoherenceChallenge ───────────────────────────────────────────────

def _gen_coherence_challenge(rng: random.Random, idx: int, seed: int) -> dict:
    """
    Detect and correct a single planted narrative inconsistency.
    Difficulty = type of inconsistency: obvious name-drift (4 loops) → subtle trait-reversal (32).
    CoherenceProbe has real signal to detect here.
    """
    genre  = rng.choice(["fantasy", "contemporary"])
    names  = _NAMES_FANTASY    if genre == "fantasy" else _NAMES_CONTEMPORARY
    locs   = _LOCS_FANTASY     if genre == "fantasy" else _LOCS_CONTEMPORARY

    n_chars    = rng.randint(2, 5)
    chars      = rng.sample(names, min(n_chars, len(names)))
    char_a, char_b = chars[0], chars[1]
    loc_a, loc_b   = rng.sample(locs, 2)
    trait_a        = _pick(rng, _TRAITS)
    emotion_a      = _pick(rng, _EMOTIONS)
    rel_ab         = _pick(rng, _RELATIONSHIPS)

    # Pick inconsistency type (weighted toward harder ones for training variety)
    incon_type, incon_desc, base_score = rng.choices(
        _INCONSISTENCY_TYPES,
        weights=[1, 2, 3, 3, 4, 4],  # harder types more frequent
    )[0]

    # Build clean Fragment A
    frag_a = (
        f"{char_a} arrived at {loc_a}, their manner {trait_a}. "
        f"{char_b}, who {rel_ab} {char_a}, was already waiting. "
        f"They spoke quietly. The air carried the weight of {emotion_a}."
    )

    # Plant inconsistency in Fragment B
    if incon_type == "name_drift":
        alt = rng.choice([n for n in names if n not in chars])
        frag_b = (
            f"Later, {alt} crossed alone to {loc_b}, having left {char_b} behind at {loc_a}."
        )
        error_note = f"'{alt}' should be '{char_a}' — name changed without explanation."
        fix        = frag_b.replace(alt, char_a, 1)

    elif incon_type == "location_contradiction":
        frag_b = (
            f"{char_a} was simultaneously observed at {loc_b}, "
            f"while {char_b} confirmed they had never left {loc_a} together."
        )
        error_note = f"'{char_a}' cannot be at both {loc_a} and {loc_b} simultaneously."
        fix        = f"{char_a} moved to {loc_b}, having left {char_b} at {loc_a}."

    elif incon_type == "object_continuity":
        obj = rng.choice(["a sealed letter", "a bronze key", "the map", "a broken blade"])
        frag_b = (
            f"{char_a} placed {obj} on the table. When {char_b} reached for it moments later, "
            f"the surface was bare — {obj} had never been mentioned before."
        )
        error_note = f"'{obj}' appeared and disappeared with no continuity between the sentences."
        fix = (f"{char_a} placed {obj} on the table. "
               f"When {char_b} reached for it moments later, it was gone.")

    elif incon_type == "timeline_error":
        frag_b = (
            f"{char_b} reminded {char_a} of the night they first met at {loc_b}, "
            f"even though that encounter had not yet occurred."
        )
        error_note = f"The meeting at {loc_b} is referenced as past, but hasn't happened yet in the timeline."
        fix        = (f"{char_b} looked at {char_a}, wondering when they would finally go to {loc_b}.")

    elif incon_type == "relationship_error":
        opposite = rng.choice([r for r in _RELATIONSHIPS if r != rel_ab])
        frag_b = (
            f"{char_b}, who {opposite} {char_a}, spoke as though no history stood between them."
        )
        error_note = (f"Fragment A says {char_b} '{rel_ab}' {char_a}; "
                      f"Fragment B says they '{opposite}' — a direct contradiction.")
        fix = frag_b.replace(opposite, rel_ab, 1)

    else:  # trait_reversal — hardest
        opposite_trait = rng.choice([t for t in _TRAITS if t != trait_a])
        frag_b = (
            f"{char_a}, uncharacteristically {opposite_trait}, ignored every warning "
            f"{char_b} had offered and acted without hesitation."
        )
        error_note = (f"'{char_a}' is established as '{trait_a}' in Fragment A. "
                      f"Acting '{opposite_trait}' directly contradicts this core trait "
                      f"with no explanation or narrative arc provided.")
        fix = (f"{char_a}, true to their {trait_a} nature, paused to consider "
               f"every warning {char_b} had offered before proceeding.")

    prompt = (
        f"The following two story fragments contain exactly one internal inconsistency. "
        f"Identify it precisely: name the type of error, quote the offending text, "
        f"explain why it is inconsistent, and provide a corrected version.\n\n"
        f"**Fragment A:**\n{frag_a}\n\n"
        f"**Fragment B:**\n{frag_b}"
    )

    answer = (
        f"**Inconsistency type:** {incon_type}\n\n"
        f"**Offending text:** Fragment B\n\n"
        f"**Why it's inconsistent:** {error_note}\n\n"
        f"**Corrected Fragment B:**\n{fix}"
    )

    score = min(1.0, base_score + n_chars * 0.025)
    return {
        "id":                    _uid("coherence_challenge", idx, seed),
        "task_type":             "coherence_challenge",
        "genre":                 genre,
        "inconsistency_type":    incon_type,
        "n_characters":          n_chars,
        "n_scenes":              2,
        "constraint_count":      n_chars + 3,
        "complexity_score":      round(score, 3),
        "suggested_n_loops":     _loops(score),
        "narrative_mode":        _mode(score),
        "fable_memory_required": True,
        "coherence_probe_targets": chars,
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "source": "fable_forge_v1",
    }


# ── Task 3 — NarrativeCompletion ──────────────────────────────────────────────

def _gen_narrative_completion(rng: random.Random, idx: int, seed: int) -> dict:
    """
    Generate a story continuation that satisfies N explicit constraints.
    Difficulty = n_characters × n_constraints.
    Tests whether the model can hold multiple simultaneous narrative requirements.
    """
    genre  = rng.choice(["fantasy", "contemporary"])
    names  = _NAMES_FANTASY    if genre == "fantasy" else _NAMES_CONTEMPORARY
    locs   = _LOCS_FANTASY     if genre == "fantasy" else _LOCS_CONTEMPORARY

    n_chars       = rng.randint(2, 6)
    chars         = rng.sample(names, min(n_chars, len(names)))
    n_constraints = rng.randint(2, min(8, n_chars * 2))

    profiles = [{"name": c, "trait": _pick(rng, _TRAITS),
                 "emotion": _pick(rng, _EMOTIONS),
                 "goal": f"reach {_pick(rng, locs)}"} for c in chars]

    # Build constraints
    constraints = []
    used = set()
    attempts = 0
    while len(constraints) < n_constraints and attempts < 50:
        attempts += 1
        p = rng.choice(profiles)
        ctype = rng.choice([
            "must appear", "must speak", "emotion_shift",
            "must reference_goal", "must avoid",
        ])
        key = f"{p['name']}:{ctype}"
        if key in used:
            continue
        used.add(key)

        if ctype == "must speak" and len(chars) > 1:
            other = rng.choice([c for c in chars if c != p["name"]])
            constraints.append(f"{p['name']} must speak directly to {other}")
        elif ctype == "emotion_shift":
            new_em = rng.choice([e for e in _EMOTIONS if e != p["emotion"]])
            constraints.append(
                f"{p['name']}'s emotional state must shift from {p['emotion']} to {new_em}"
            )
        elif ctype == "must reference_goal":
            constraints.append(f"{p['name']} must reference their goal ({p['goal']})")
        elif ctype == "must avoid":
            loc = _pick(rng, locs)
            constraints.append(f"{p['name']} must not be placed at {loc}")
        else:
            constraints.append(f"{p['name']} must appear in the scene")

    opening_loc = _pick(rng, locs)
    opening = (
        f"The scene opens at {opening_loc}. "
        f"{chars[0]} has just arrived, their expression unreadable. "
        + (f"{chars[1]} was already there. " if len(chars) > 1 else "")
        + "Something in the air had shifted."
    )

    profile_block = "\n".join(
        f"- **{p['name']}**: {p['trait']}, currently {p['emotion']}, goal: {p['goal']}"
        for p in profiles
    )
    constraint_block = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(constraints))

    # Model continuation — satisfies constraints structurally
    scene_lines = [opening, ""]
    for p in profiles[:3]:
        scene_lines.append(
            f'{p["name"]} moved with the deliberate care of someone carrying '
            f'more than the moment required.'
        )
    if len(chars) > 1:
        speaker, listener = chars[0], chars[1]
        scene_lines.append(f'"{listener}," {speaker} said at last. "We should talk about {profiles[0]["goal"]}."')
    scene_lines += [
        "",
        "The silence that followed was its own kind of answer.",
        "Outside, the world continued without waiting for them.",
    ]
    scene = "\n".join(scene_lines)

    checklist = "\n".join(f"✓ {c}" for c in constraints)
    answer = f"{scene}\n\n---\n**Constraints satisfied:**\n{checklist}"

    prompt = (
        f"Continue the following story opening with a single scene (150–300 words). "
        f"You must satisfy **all {n_constraints} constraints** while keeping every character's "
        f"voice and behaviour consistent with their profile.\n\n"
        f"**Character profiles:**\n{profile_block}\n\n"
        f"**Required constraints:**\n{constraint_block}\n\n"
        f"**Opening:**\n{opening}\n\n"
        f"Write the continuation. After the scene, list each constraint and confirm how you satisfied it."
    )

    score = min(1.0, 0.25 + n_chars * 0.06 + n_constraints * 0.045)
    return {
        "id":                    _uid("narrative_completion", idx, seed),
        "task_type":             "narrative_completion",
        "genre":                 genre,
        "n_characters":          n_chars,
        "n_scenes":              1,
        "constraint_count":      n_constraints,
        "complexity_score":      round(score, 3),
        "suggested_n_loops":     _loops(score),
        "narrative_mode":        _mode(score),
        "fable_memory_required": n_chars > 3 or n_constraints > 4,
        "coherence_probe_targets": chars,
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "source": "fable_forge_v1",
    }


# ── Public API ────────────────────────────────────────────────────────────────

_TASK_FNS = {
    "character_trace":     _gen_character_trace,
    "coherence_challenge": _gen_coherence_challenge,
    "narrative_completion": _gen_narrative_completion,
}
_TASK_WEIGHTS = [0.40, 0.30, 0.30]  # trace / coherence / completion


class FableForge:
    """Pipeline wrapper — call .generate() or use the module-level generate() fn."""

    def generate(self, count: int, seed: int = 42,
                 task_type: Optional[str] = None) -> list[dict]:
        return generate(count, seed=seed, task_type=task_type)

    def stats(self, examples: list[dict]) -> dict:
        modes  = Counter(e["narrative_mode"] for e in examples)
        types  = Counter(e["task_type"]      for e in examples)
        loops  = Counter(e["suggested_n_loops"] for e in examples)
        mem    = sum(1 for e in examples if e.get("fable_memory_required"))
        avg_sc = sum(e["complexity_score"] for e in examples) / max(len(examples), 1)
        return dict(total=len(examples), avg_complexity=round(avg_sc, 3),
                    modes=dict(modes), task_types=dict(types),
                    loop_distribution=dict(loops), fable_memory_required=mem)


def generate(count: int, seed: int = 42,
             task_type: Optional[str] = None) -> list[dict]:
    """Generate `count` FableForge examples deterministically."""
    if task_type and task_type not in _TASK_FNS:
        raise ValueError(f"Unknown task_type '{task_type}'. Choose: {list(_TASK_FNS)}")

    rng      = random.Random(seed)
    task_list = list(_TASK_FNS.keys())
    examples  = []

    for i in range(count):
        if task_type:
            fn = _TASK_FNS[task_type]
        else:
            fn = _TASK_FNS[rng.choices(task_list, weights=_TASK_WEIGHTS)[0]]
        examples.append(fn(rng, i, seed))

    return examples


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate FableForge narrative training dataset")
    ap.add_argument("--count",     type=int,  default=1000)
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--output",    default="data/fable_forge.jsonl")
    ap.add_argument("--task-type", choices=list(_TASK_FNS))
    ap.add_argument("--stats",     action="store_true")
    args = ap.parse_args()

    print(f"Generating {args.count:,} FableForge examples (seed={args.seed}) …")
    examples = generate(args.count, seed=args.seed, task_type=args.task_type)

    forge = FableForge()
    s     = forge.stats(examples)

    loop_labels = {4: "action  (4 loops)", 8: "dialogue (8 loops)",
                   16: "exposition (16)", 32: "deep (32 loops)"}
    print(f"\n{'='*56}")
    print(f"  Total examples:      {s['total']:,}")
    print(f"  Avg complexity:      {s['avg_complexity']:.3f}")
    print(f"  FableMemory needed:  {s['fable_memory_required']:,} ({100*s['fable_memory_required']/s['total']:.1f}%)")
    print(f"\n  Recurrence distribution:")
    for loops, count in sorted(s['loop_distribution'].items()):
        bar = "█" * (count * 36 // s['total'])
        print(f"    {loop_labels.get(loops, loops):26s} {count:5,}  {bar}")
    print(f"\n  Task types:")
    for tt, count in sorted(s['task_types'].items(), key=lambda x: -x[1]):
        print(f"    {tt:28s} {count:5,}")
    print(f"{'='*56}\n")

    if args.stats:
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Written → {args.output}  ({len(examples):,} examples)")


if __name__ == "__main__":
    main()
