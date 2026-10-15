"""Convert flat ambiguous_arc.json items to the experiment format (AmbigousARC.json).

Usage: python convert_items.py ambiguous_arc.json AmbigousARC.json [--seed 42] [--n_options 4]
"""
import argparse
import hashlib
import json
import random

import numpy as np

ROW_CONCEPTS = {"keep_above_or_below"}


def to_array(s, dim):
    return np.array([int(ch) for ch in s]).reshape(dim, dim)


def encode(arr):
    return "".join(str(x) for x in arr.flatten())


def item_seed(item_id, seed):
    """Per-item seed (same scheme as generate_seed in prompt_utils.py)."""
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    return (h + seed) % (2**32 - 1)


def d_matrix(item, level="pixel"):
    """Where A and B agree, copy C; where they differ, copy B (per cell or per row)."""
    if level == "pixel":
        return "".join(c if a == b else b for a, b, c in zip(item["A"], item["B"], item["C"]))
    dim = item["xdim"]
    A, B, C = (to_array(item[k], dim) for k in "ABC")
    return "".join(encode(c if np.array_equal(a, b) else b) for a, b, c in zip(A, B, C))


def d_random(item, rng, exclude, max_iter=10_000):
    """Grid built from random rows/columns of A, B, C; must differ from all grids in `exclude`."""
    dim = item["xdim"]
    inputs = [to_array(item[k], dim) for k in "ABC"]
    for _ in range(max_iter):
        m = np.empty((dim, dim), dtype=int)
        for i in range(dim):
            m[:, i] = inputs[rng.integers(3)][:, rng.integers(dim)]
            m[i, :] = inputs[rng.integers(3)][rng.integers(dim), :]
        s = encode(m)
        if s not in exclude:
            return s
    raise RuntimeError(f"No distinct D_Random found for {item['id']}")


def convert(items, seed=42, n_options=4):
    all_concepts = sorted({it["concept"] for it in items})
    if n_options > len(all_concepts):
        raise ValueError(f"n_options={n_options} but only {len(all_concepts)} concepts in file")

    out = []
    for it in items:
        s = item_seed(it["id"], seed)
        rng = np.random.default_rng(s)
        prng = random.Random(s)

        dm = d_matrix(it, "pixel")
        dm_row = d_matrix(it, "row") if it["concept"] in ROW_CONCEPTS else None
        exclude = {it["A"], it["B"], it["C"], it["D_Concept"], dm} | ({dm_row} if dm_row else set())

        distractors = prng.sample([c for c in all_concepts if c != it["concept"]], n_options - 1)
        options = distractors + [it["concept"]]
        prng.shuffle(options)

        out.append({
            "main_id": it["id"],
            "item_id": f"{it['id']}_1",
            "mirror": "1",
            "A": it["A"],
            "B": it["B"],
            "C": it["C"],
            "D_Concept": it["D_Concept"],
            "D_Matrix": dm,
            "D_Matrix_Row": dm_row,
            "D_Random": d_random(it, rng, exclude),
            "xdim": it["xdim"],
            "ydim": it["ydim"],
            "concept": it["concept"],
            "concept_options": options,
        })
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("infile")
    p.add_argument("outfile")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_options", type=int, default=4)
    args = p.parse_args()

    with open(args.infile) as f:
        items = json.load(f)
    out = convert(items, args.seed, args.n_options)
    with open(args.outfile, "w") as f:
        json.dump(out, f, indent=4)
    print(f"Wrote {len(out)} items to {args.outfile}")