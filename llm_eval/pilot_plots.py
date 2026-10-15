"""
Quick plots for scored LLM runs.

    from score import score_run
    from pilot_plots import plot_performance, plot_item_across_models, plot_responses

    scored = score_run(Path(LLM_RESULTS_DIR) / "raw" / "pilot.jsonl")
    plot_performance(scored)
    plot_item_across_models(scored, "02ue5B_1")
    plot_responses(scored, model="openai/gpt-4o", category="other", n=3)
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from utils.plot_utils import plot_matrix, plot_item          # collaborator's styling
from utils.prompt_utils import convert_to_array
from utils.globals import ITEMS_FILE

from score import CATEGORIES

CAT_COLORS = {
    "concept": "#2b8a3e", "literal": "#1971c2", "duplicate": "#e8590c",
    "other": "#8242b7", "unparseable": "#ced4da", "truncated": "#d6cc0c",
    "no_response": "#dee2e6", "error": "#c92a2a",
}


def _load_items_and_answers(prompts_path, items_path):
    prompts = json.load(open(prompts_path))
    answers = dict(zip(prompts["item_ids"], prompts["answers"]))   # [D_Literal, D_Concept]
    items = {i["id"]: i for i in json.load(open(items_path))}
    return items, answers


def _short(model):
    return model.split("/")[-1]


def plot_performance(scored, sort_by="concept", ax=None):
    """Stacked horizontal bars: share of each response category per model."""
    props = (scored.groupby("model", observed=True)["category"]
                   .value_counts(normalize=True).unstack(fill_value=0))
    props = props.reindex(columns=[c for c in CATEGORIES if c in props.columns])
    props = props.sort_values(sort_by, ascending=True) if sort_by in props else props
    n = scored.groupby("model").size().reindex(props.index)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.45 * len(props) + 1.2))
    left = np.zeros(len(props))
    for cat in props.columns:
        ax.barh(range(len(props)), props[cat], left=left,
                color=CAT_COLORS[cat], label=cat, edgecolor="white", linewidth=0.5)
        left += props[cat].values
    ax.set_yticks(range(len(props)))
    ax.set_yticklabels([f"{_short(m)}  (n={n[m]})" for m in props.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of responses")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax


def plot_item_across_models(scored, item_id,
                            prompts_path=HERE / "prompts.json",
                            items_path=ITEMS_FILE,
                            ncols=5):
    """Top row: A, B, C, D Concept, D Literal. Below: each model's response."""
    items, answers = _load_items_and_answers(prompts_path, items_path)
    item = items[item_id.split("_")[0]]
    d = item["xdim"]
    d_literal, d_concept = answers[item_id]
    sub = scored[scored.item_id == item_id].sort_values("model")

    nrows = 1 + int(np.ceil(len(sub) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.7 * nrows))
    axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")

    top = [("A", item["A"]), ("B", item["B"]), ("C", item["C"]),
           ("D Concept", d_concept), ("D Literal", d_literal)]
    for ax, (label, grid) in zip(axs[0], top):
        ax.axis("on")
        plot_matrix(convert_to_array(grid, d), label, ax=ax, size=10)

    for ax, (_, r) in zip(axs[1:].flat, sub.iterrows()):
        ax.axis("on")
        if isinstance(r["parsed"], str):
            plot_matrix(convert_to_array(r["parsed"], d), "", ax=ax)
        else:
            ax.imshow(np.full((d, d), np.nan))
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "no grid", ha="center", va="center",
                    transform=ax.transAxes, color="grey", size=9)
        for spine in ax.spines.values():                 # category shown as border colour
            spine.set_edgecolor(CAT_COLORS.get(r["category"], "black"))
            spine.set_linewidth(3)
        ax.set_title(f"{_short(r['model'])}\n{r['category']}", size=8)

    fig.suptitle(f"{item_id} | {item['concept']}", size=12)
    plt.tight_layout()
    plt.show()


def plot_responses(scored, model=None, category=None, item_ids=None, n=5,
                   prompts_path=HERE / "prompts.json",
                   items_path=ITEMS_FILE):
    """Collaborator-style item plots (A->B, C->Response, D Concept, D Literal), filtered."""
    items, answers = _load_items_and_answers(prompts_path, items_path)
    sub = scored
    if model is not None:
        sub = sub[sub.model == model]
    if category is not None:
        sub = sub[sub.category == category]
    if item_ids is not None:
        sub = sub[sub.item_id.isin(item_ids)]

    for _, r in sub.head(n).iterrows():
        item = items[r["item_main_id"]]
        d = item["xdim"]
        d_literal, d_concept = answers[r["item_id"]]
        if isinstance(r["parsed"], str):
            resp, resp_label = convert_to_array(r["parsed"], d), "Response"
        else:
            resp, resp_label = np.zeros((d, d), dtype=int), r["category"].capitalize()
            print(f"{r['item_id']} | {_short(r['model'])} raw: {r['raw']!r}")
        arrs = [convert_to_array(item[k], d) for k in ["A", "B", "C"]] + \
               [resp, convert_to_array(d_concept, d), convert_to_array(d_literal, d)]
        plot_item(arrays=arrs,
                  title=f"{_short(r['model'])} | {r['item_id']} | {item['concept']} | {r['category']}",
                  labels=["A", "B", "C", resp_label, "D Concept", "D Literal"])