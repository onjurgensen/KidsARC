"""
Score raw OpenRouter results (results/raw/<run>.jsonl) into a long-format table.

One row per (model, item). If an item was queried more than once (e.g. retried
after an error), the LAST row in the file is used.

Categories (in order of precedence):
    error        API call failed after retries
    no_response  empty text
    truncated    hit max_tokens (finish_reason == "length") and no parseable grid
    unparseable  text present, but no dim x dim grid could be extracted
    concept      == D_Concept
    matrix       == D_Matrix
    duplicate    == A, B or C (collaborator's definition)
    other        valid grid, none of the above

Raw and scored runs live under LLM_RESULTS_DIR (globals.yml).

Usage:
    python score.py <LLM_RESULTS_DIR>/raw/pilot.jsonl   # writes <LLM_RESULTS_DIR>/scored/pilot.csv
    python score.py <raw.jsonl> --out other.csv          # write somewhere else
or from a notebook:
    from score import score_run
    scored = score_run(Path(LLM_RESULTS_DIR) / "raw" / "pilot.jsonl")

When reading the CSV back, use dtype=str for grid columns (leading zeros!).
"""
import re
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from utils.globals import ITEMS_FILE, LLM_RESULTS_DIR

CATEGORIES = ["concept", "matrix", "duplicate", "other",
              "unparseable", "truncated", "no_response", "error"]


def parse_grid(text, dim):
    """Return the response as a dim*dim digit string, or None if unparseable."""
    if text is None:
        return None
    rows = ["".join(re.findall(r"\d", r)) for r in re.findall(r"\[([^\[\]]*)\]", text)]
    rows = [r for r in rows if r]
    for start in range(len(rows) - dim + 1):        # first `dim` consecutive rows of length `dim`
        block = rows[start:start + dim]
        if all(len(r) == dim for r in block):
            return "".join(block)
    digits = "".join(re.findall(r"\d", text))       # fallback: no usable brackets
    return digits if len(digits) == dim * dim else None


def _missing(x):
    return x is None or (isinstance(x, float) and pd.isna(x))


def categorize(row, item, d_matrix, d_concept):
    """Return (category, parsed_grid)."""
    if not _missing(row.get("error")):
        return "error", None
    text = row.get("text")
    truncated = row.get("finish_reason") == "length"
    if _missing(text) or not str(text).strip():
        return ("truncated" if truncated else "no_response"), None
    grid = parse_grid(text, item["xdim"])
    if grid is None:
        return ("truncated" if truncated else "unparseable"), None
    if grid == d_concept:
        return "concept", grid
    if grid == d_matrix:
        return "matrix", grid
    if grid in (item["A"], item["B"], item["C"]):
        return "duplicate", grid
    return "other", grid


def load_raw(path):
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["model", "item_id"], keep="last").reset_index(drop=True)


def score_run(raw_path,
              prompts_path=HERE / "prompts.json",
              items_path=ITEMS_FILE):
    raw = load_raw(raw_path)
    prompts = json.load(open(prompts_path))
    answers = dict(zip(prompts["item_ids"], prompts["answers"]))   # item_id -> [D_Matrix, D_Concept]
    items = {i["id"]: i for i in json.load(open(items_path))}

    out = []
    for _, row in raw.iterrows():
        row = row.to_dict()
        main_id = row["item_id"].split("_")[0]
        item = items[main_id]
        d_matrix, d_concept = answers[row["item_id"]]
        cat, grid = categorize(row, item, d_matrix, d_concept)
        out.append({
            "model": row["model"],
            "item_id": row["item_id"],
            "item_main_id": main_id,
            "concept": item["concept"],
            "dim": item["xdim"],
            "category": cat,
            "parsed": grid,
            "raw": None if _missing(row.get("text")) else row.get("text"),
            "finish_reason": row.get("finish_reason"),
            "provider_served": row.get("provider_served"),
            "reasoning_tokens": row.get("reasoning_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "cost": row.get("cost"),
            "rt": row.get("rt"),
            # one-hot columns, matching the human data / R preprocessing
            "concept_response": int(cat == "concept"),
            "matrix_response": int(cat == "matrix"),
            "duplicate_response": int(cat == "duplicate"),
        })
    df = pd.DataFrame(out)
    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_path", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <LLM_RESULTS_DIR>/scored/<run>.csv")
    args = ap.parse_args()

    scored = score_run(args.raw_path)
    out_path = args.out or Path(LLM_RESULTS_DIR) / "scored" / f"{args.raw_path.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(scored.groupby("model", observed=True)["category"]
                .value_counts().unstack(fill_value=0))
    print(f"\nWrote {out_path}")