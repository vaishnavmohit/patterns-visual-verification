"""Item inventory and factor definitions for the factorial experiment.

The 200 queries are the same items used in the human study: 50 from each of
the four Bongard-LOGO family JSONs (bd, ff, hd_comb, hd_novel). Every item
carries three oracle representations of the same 13 figures (6 positive,
6 negative, 1 query):

* ``programs``     raw action-program strings (Action Programs interface)
* ``descriptions`` drawing stories derived from the programs (Action
                   Descriptions interface; regenerated here with the same
                   function the archived runs used)
* ``images``       rendered PNG paths (Direct Visual interface)

The wrong-cue mapping is a deterministic cyclic derangement over the distinct
concept labels within each family, so every wrong cue is a real label from the
same family that never equals the item's own label.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..datasets.action_descriptions import process_all_image_specs, process_all_image_specs_rendered

FAMILIES = ("bd", "ff", "hd_comb", "hd_novel")
FAMILY_LABELS = {
    "bd": "Basic",
    "ff": "Free-form",
    "hd_comb": "Human-defined",
    "hd_novel": "Human-defined Novel",
}

REPRESENTATIONS = ("pixels", "caption_self", "caption_strong", "oracle_description", "program",
                   # Open-weight replication (docs/factorial_openweight_plan.md):
                   "oracle_description_rendered",  # descriptions that follow the renderer's geometry
                   "caption_self_free",            # own captions in free prose rather than the step recipe
                   # Amendments A3/A4: the same two caption protocols from independent sampled draws.
                   "caption_self_d2", "caption_self_free_d2",
                   "caption_self_d3", "caption_self_free_d3",
                   "caption_self_d4", "caption_self_free_d4")
# Additional caption draws (amendments A3, A4): representation -> (caption style, draw name).
EXTRA_DRAWS = ("d2", "d3", "d4")
DRAW_REPRESENTATIONS = {**{f"caption_self_{d}": ("steps", d) for d in EXTRA_DRAWS},
                        **{f"caption_self_free_{d}": ("free", d) for d in EXTRA_DRAWS}}
SELF_CAPTION_REPRESENTATIONS = ("caption_self", "caption_self_free", *DRAW_REPRESENTATIONS)
CUES = ("none", "correct", "wrong",
        # Open-weight replication: a mismatched label sharing no component with the
        # item's own label, and length-matched text with no geometric content.
        "wrong_clean", "irrelevant")
WRONG_CUE_SEED = 20260907
M_POS = 6
N_NEG = 6


@dataclass
class Item:
    test_id: str
    uid: str
    family: str
    polarity: str            # 'pos' or 'neg' (ground-truth query label)
    concept: str             # human concept label shown in the cue condition
    caption: str             # raw benchmark caption (e.g. 'bd_...')
    programs: list[list[str]]
    images: list[str]        # relative paths, len 13
    descriptions: list[str] = field(default_factory=list)
    descriptions_rendered: list[str] = field(default_factory=list)

    @property
    def base_id(self) -> str:
        return self.uid

    def image_abs(self, image_dir: str | Path) -> list[str]:
        return [str(Path(image_dir) / p) for p in self.images]


def load_items(data_dir: str | Path, families: Iterable[str] = FAMILIES) -> list[Item]:
    """Load all items, sorted by family then test_id for a stable order."""
    items: list[Item] = []
    for fam in families:
        path = Path(data_dir) / "Bongard_Logo_human" / f"{fam}.json"
        with open(path) as f:
            data = json.load(f)
        for s in data:
            programs = s["action_prog"]["pos"][:M_POS] + s["action_prog"]["neg"][:N_NEG] + [s["testfiles"]["action_prog"]]
            images = s["images"]["pos"][:M_POS] + s["images"]["neg"][:N_NEG] + [s["images"]["testfiles"]]
            if len(programs) != M_POS + N_NEG + 1 or len(images) != M_POS + N_NEG + 1:
                raise ValueError(f"{s['test_id']}: expected 13 figures, got {len(programs)} programs / {len(images)} images")
            item = Item(
                test_id=s["test_id"],
                uid=s["uid"],
                family=fam,
                polarity=s["testfiles"]["category"],
                concept=s["concept_ui"],
                caption=s["caption"],
                programs=programs,
                images=images,
            )
            item.descriptions = process_all_image_specs([list(p) for p in programs])
            item.descriptions_rendered = process_all_image_specs_rendered([list(p) for p in programs])
            items.append(item)
    items.sort(key=lambda it: (FAMILIES.index(it.family), it.test_id))
    return items


def pilot_subset(items: list[Item], per_family: int) -> list[Item]:
    """First ``per_family`` items of each family (stable order)."""
    out: list[Item] = []
    for fam in FAMILIES:
        out.extend([it for it in items if it.family == fam][:per_family])
    return out


# ---------------------------------------------------------------------------
# Wrong-cue mapping
# ---------------------------------------------------------------------------

def build_wrong_cue_map(items: list[Item], seed: int = WRONG_CUE_SEED) -> dict[str, str]:
    """Map each concept label to a different label from the same family.

    A seeded shuffle of the distinct labels within a family followed by a
    cyclic shift gives a derangement: label_i -> label_{i+1}. Families with a
    single distinct label cannot receive a wrong cue and raise.
    """
    rng = random.Random(seed)
    mapping: dict[str, str] = {}
    for fam in FAMILIES:
        labels = sorted({it.concept for it in items if it.family == fam})
        if len(labels) < 2:
            raise ValueError(f"family {fam} has fewer than two distinct concept labels")
        rng.shuffle(labels)
        for i, lab in enumerate(labels):
            mapping[f"{fam}::{lab}"] = labels[(i + 1) % len(labels)]
    return mapping


def wrong_cue_for(item: Item, mapping: dict[str, str]) -> str:
    return mapping[f"{item.family}::{item.concept}"]


def write_wrong_cue_file(items: list[Item], path: str | Path, seed: int = WRONG_CUE_SEED) -> dict[str, str]:
    mapping = build_wrong_cue_map(items, seed)
    payload = {
        "seed": seed,
        "method": "seeded shuffle of distinct labels within family, cyclic shift by one",
        "families": {},
    }
    for fam in FAMILIES:
        payload["families"][fam] = {
            k.split("::", 1)[1]: v for k, v in mapping.items() if k.startswith(f"{fam}::")
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return mapping


def load_wrong_cue_file(path: str | Path) -> dict[str, str]:
    with open(path) as f:
        payload = json.load(f)
    mapping: dict[str, str] = {}
    for fam, table in payload["families"].items():
        for k, v in table.items():
            mapping[f"{fam}::{k}"] = v
    return mapping


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    name: str
    temperature: float
    seed: int


@dataclass(frozen=True)
class Cell:
    model: str
    representation: str
    cue: str
    run: RunSpec

    @property
    def context(self) -> str:
        """bongard_context string understood by hmr.prompts.system_eval_prompt."""
        base = {
            "pixels": "dvrl",
            "caption_self": "action_description",
            "caption_strong": "action_description",
            "oracle_description": "action_description",
            "oracle_description_rendered": "action_description",
            "caption_self_free": "free_description",
            **{rep: ("action_description" if style == "steps" else "free_description")
               for rep, (style, _) in DRAW_REPRESENTATIONS.items()},
            "program": "aps",
        }[self.representation]
        return base if self.cue == "none" else f"{base}_concept"

    @property
    def needs_images(self) -> bool:
        return self.representation == "pixels"

    @property
    def path_parts(self) -> tuple[str, str, str, str]:
        return (slug(self.model), self.representation, self.cue, self.run.name)


def slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def enumerate_cells(
    model: str,
    model_cfg: dict,
    representations: Iterable[str],
    cues: Iterable[str],
    runs: Iterable[RunSpec],
    strong_captioner: str,
) -> tuple[list[Cell], list[str]]:
    """Return the cells to run for one model plus human-readable skip reasons."""
    cells: list[Cell] = []
    skipped: list[str] = []
    vision = bool(model_cfg.get("vision", True))
    supports_temp = bool(model_cfg.get("supports_temperature", True))
    # A vision model may still opt out of writing its own captions (provider
    # rate limits make 2,504 single-image calls impractical). It then receives
    # the strong captioner's descriptions through caption_strong instead, and
    # its perception arm is reported as externally captioned.
    self_captions = bool(model_cfg.get("self_captions", True))
    for rep in representations:
        if rep not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {rep!r}")
        if (rep == "pixels" or rep in SELF_CAPTION_REPRESENTATIONS) and not vision:
            skipped.append(f"{rep}: model is text-only")
            continue
        if rep in SELF_CAPTION_REPRESENTATIONS and not self_captions:
            skipped.append("caption_self: disabled for this model; caption_strong supplies its descriptions")
            continue
        if rep == "caption_strong" and model == strong_captioner:
            skipped.append("caption_strong: identical to caption_self for the strong captioner")
            continue
        for cue in cues:
            if cue not in CUES:
                raise ValueError(f"unknown cue {cue!r}")
            for run in runs:
                if run.temperature == 0.0 and not supports_temp:
                    skipped.append(f"{rep}/{cue}/{run.name}: provider fixes temperature")
                    continue
                cells.append(Cell(model, rep, cue, run))
    return cells, sorted(set(skipped))
