"""Extend the general-domain prompt battery to 256+ tokens.

exp256's final-number run needs >192 tokens for train-192; the
original battery has 86. Appends general prose (news, science,
history, conversation — NOT code) and writes
outputs/cluster_prompts_256.json.

Usage:
    python scripts/v4_make_general256.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

TOKENIZER_JSON = ROOT / "outputs" / "v4_tokenizer.json"
BASE = ROOT / "outputs" / "cluster_prompts.json"

GENERAL_PROSE = [
    "The history of the steam engine begins with ancient experiments on steam power, but the practical era opened in the early eighteenth century.",
    "Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight absorbed by chlorophyll.",
    "The Great Barrier Reef is the largest living structure on Earth and is visible from space.",
    "Classical mechanics describes the motion of bodies under the influence of forces, with Newton's laws forming its foundation.",
    "The Industrial Revolution transformed agrarian societies into industrial ones through mechanization and factory production.",
    "Volcanic eruptions can affect global climate by injecting sulfate aerosols into the stratosphere.",
    "The study of linguistics examines how languages evolve, how they are structured, and how they are acquired by children.",
    "Ocean currents distribute heat around the planet, with the Gulf Stream warming the climate of western Europe.",
    "Ancient Egyptian civilization developed along the Nile river over five thousand years ago.",
    "The immune system defends the body against pathogens through a network of cells and signaling molecules.",
    "Renaissance painters developed linear perspective to create the illusion of depth on flat surfaces.",
    "The theory of plate tectonics explains earthquakes, mountain building, and the drifting of continents.",
    "Coffee was first cultivated in Ethiopia and spread through the Arab world before reaching Europe.",
    "Statistical mechanics links the microscopic behavior of particles to macroscopic thermodynamic quantities.",
    "The Roman road network once spanned over four hundred thousand kilometers across three continents.",
]


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    base = json.load(open(BASE, encoding="utf-8"))
    ids = [int(i) for i in base["token_ids"]]
    n_added = 0
    for p in GENERAL_PROSE:
        enc = tok.encode(p)
        if enc.ids:
            ids.extend(int(i) for i in enc.ids)
            n_added += 1
    out = {"token_ids": ids, "n": len(ids),
           "n_base": len(base["token_ids"]), "n_prompts_added": n_added}
    dest = ROOT / "outputs" / "cluster_prompts_256.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}: {len(ids)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
