"""
Visualisation of individual responses, for humans and models alike.

Takes a row from either `Eval(...).df` (models) or `HumanEval(...).df`
(participants) and draws the item together with what was answered.

    from utils.eval import Eval
    from utils.human_eval import HumanEval
    from utils.prompt_utils import AmbigousARCDataset
    from utils.visualize_responses import ResponseBrowser, plot_response

    # Models
    disc = Eval('data/results/run_2024-11-12/discrimination.json')
    browser = ResponseBrowser(disc.df, disc.dataset, item_id='02ue5B_1')
    browser.show(); browser.next()

    # Humans
    human = HumanEval('path/to/responses.csv', task='discrimination')
    browser = ResponseBrowser(human.df, 'path/to/items.json', item_id='02ue5B_1')
    browser.show()

    # Or a single row directly
    plot_response(human.df.iloc[0], items)

Grid drawing is delegated to `utils.plot_utils`, so the colours match every
other figure in the repo.
"""

import json
import warnings
from typing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from utils.plot_utils import plot_matrix, ARC_cmap
from utils.prompt_utils import convert_to_array

# Columns that identify the respondent, in order of preference.
SUBJECT_COLS = ['participant', 'model', 'participant_fk']

# Candidate answers, in the order they are drawn.
OPTION_MATRICES = ['D_Matrix', 'D_Concept', 'D_Random']

# `choice` -> which matrix that choice corresponds to. Shared by both frames:
# `Eval` writes matrix/concept/other/duplicate, and `HumanEval` matches it.
CHOICE_TO_MATRIX = {
    'matrix': 'D_Matrix',
    'concept': 'D_Concept',
    'duplicate': 'C',
    'other': 'D_Random',
}


# -----------------------------------------------------------------------------
# Items
# -----------------------------------------------------------------------------

def items_to_dict(items: Any) -> Dict[str, Dict]:
    """
    Normalise an item source into {item_id: item dict}.

    Accepts a path to a JSON file, a list of item dicts, or an
    `AmbigousARCDataset`. Dataset items are keyed by `id` rather than
    `item_id`, and their recognition options live on the dataset, so both
    are folded in here.
    """
    # A dataset: rebuild the records from its items_data
    if hasattr(items, 'items_data'):
        dataset = items
        records = {}
        for idx, item in enumerate(dataset.items_data):
            record = dict(item)
            record['item_id'] = str(item.get('item_id', item.get('id')))
            record['main_id'] = record['item_id'].rsplit('_', 1)[0]
            if getattr(dataset, 'concepts_options', None):
                record['concept_options'] = dataset.concepts_options[idx]
            records[record['item_id']] = record
        return records

    if isinstance(items, str):
        items = json.load(open(items, 'rb'))

    records = {}
    for item in items:
        record = dict(item)
        record['item_id'] = str(item.get('item_id', item.get('id')))
        record.setdefault('main_id', record['item_id'].rsplit('_', 1)[0])
        records[record['item_id']] = record
    return records


def get_item(items: Dict[str, Dict], item_id: str) -> Dict:
    """
    Look up an item by its versioned id, falling back to the base id.

    The fallback matters because model results and human exports do not
    always agree on whether the `_n` suffix is present.
    """
    item_id = str(item_id)
    if item_id in items:
        return items[item_id]

    base = item_id.rsplit('_', 1)[0]
    for candidate in items.values():
        if candidate.get('main_id') == base or candidate['item_id'] == base:
            return candidate

    raise KeyError(f"No item found for '{item_id}' (base id '{base}').")


def matrix_target(item: Dict, name: str, matrix_level: Any = None) -> Optional[str]:
    """The named matrix, respecting the row/pixel condition for D_Matrix."""
    if name == 'D_Matrix' and str(matrix_level).lower() == 'row':
        return item.get('D_Matrix_Row', item.get('D_Matrix'))
    return item.get(name)


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------

def _format_concept(concept: Any) -> str:
    """Match the label formatting used in the recognition prompts."""
    if concept is None or (isinstance(concept, float) and np.isnan(concept)):
        return 'No response'
    return ' '.join(str(concept).capitalize().split('_'))


def _is_blank(matrix_string: Any) -> bool:
    """An all-zero grid is a submitted-but-empty canvas, not an answer."""
    if matrix_string is None:
        return False
    s = str(matrix_string)
    return s.isdigit() and set(s) == {'0'}


def _draw_grid(ax, matrix_string, title=None, xdim=None, ydim=None):
    """Draw one grid, or a labelled blank panel if there is nothing to draw."""
    if matrix_string is None:
        ax.axis('off')
        if title:
            ax.set_title(title, size=10, color='dimgray')
        return

    digits = str(matrix_string)
    if xdim is None:
        xdim = int(round(len(digits) ** 0.5))
    if ydim is None:
        ydim = len(digits) // xdim

    # Validate before reshaping. Response grids are not guaranteed to match
    # the item's dimensions in raw exports.
    if len(digits) != xdim * ydim:
        warnings.warn(
            f'{title or "matrix"}: length {len(digits)} does not fit '
            f'{xdim} x {ydim}; panel left blank.'
        )
        ax.axis('off')
        ax.set_title(f'{title}\n(malformed)', size=10, color='firebrick')
        return

    plot_matrix(convert_to_array(digits, xdim), title=title or '', ax=ax, size=11)


def _draw_arrow(ax):
    ax.axis('off')
    ax.annotate(
        '', xy=(0.9, 0.5), xytext=(0.1, 0.5),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', linewidth=2),
    )


def _draw_text_panel(ax, heading, body, colour='black'):
    ax.axis('off')
    ax.text(0.5, 0.70, heading, ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(0.5, 0.42, body, ha='center', va='center',
            fontsize=15, fontweight='bold', color=colour, wrap=True)


def _draw_concept_options(ax, concept_options, selected, correct):
    """Recognition answer options as colour-coded boxes."""
    ax.axis('off')
    ax.set_title('Answer options', fontsize=12, fontweight='bold', pad=10)

    if not concept_options:
        ax.text(0.5, 0.5, 'No options recorded.', ha='center', va='center',
                fontsize=11, color='dimgray')
        return

    n_cols = 2
    n_rows = int(np.ceil(len(concept_options) / n_cols))
    box_width, box_height = 0.42, min(0.32, 0.78 / n_rows)
    x_positions = [0.06, 0.52]

    for index, option in enumerate(concept_options):
        x = x_positions[index % n_cols]
        y = 0.88 - (index // n_cols) * (box_height + 0.08) - box_height

        is_correct = option == correct
        is_selected = option == selected

        face, edge, lw, text_colour = 'whitesmoke', 'lightgray', 1.5, 'black'
        if is_correct and is_selected:
            face, edge, lw, text_colour = '#c7e9c0', 'forestgreen', 3.0, 'forestgreen'
        elif is_correct:
            face, edge, lw, text_colour = '#e5f5e0', 'forestgreen', 2.0, 'forestgreen'
        elif is_selected:
            face, edge, lw, text_colour = '#fde0dd', 'firebrick', 3.0, 'firebrick'

        ax.add_patch(FancyBboxPatch(
            (x, y), box_width, box_height, boxstyle='round,pad=0.02',
            linewidth=lw, edgecolor=edge, facecolor=face, transform=ax.transAxes,
        ))
        ax.text(
            x + box_width / 2, y + box_height / 2, _format_concept(option),
            ha='center', va='center', fontsize=10,
            fontweight='bold' if (is_selected or is_correct) else 'normal',
            color=text_colour, transform=ax.transAxes, wrap=True,
        )


# -----------------------------------------------------------------------------
# Resolving a response into something drawable
# -----------------------------------------------------------------------------

def resolve_response(row: pd.Series, item: Dict, task: str = None) -> Tuple[Optional[str], str]:
    """
    Return (matrix string to draw, status label) for a generation or
    discrimination response.

    The task decides where the answer lives, and getting this wrong is silent
    rather than loud, so it is checked first:

    - Generation produces a grid. Draw exactly what was submitted, never a
      canonical matrix -- an "other" response is a grid nobody offered, and
      substituting `D_Random` for it would be a fabrication.
    - Discrimination selects one of the offered matrices. `Eval` records the
      answer letter in `response` and the category in `choice`; `HumanEval`
      records the category label. Either way the category names the matrix.
    """
    matrix_level = row.get('matrix_level')

    if row.get('no_response'):
        return None, 'No response'

    response = row.get('response')
    is_missing = response is None or (
        isinstance(response, float) and pd.isna(response)
    )

    # -- Generation: the response *is* the grid -------------------------------
    if task == 'generation':
        if is_missing:
            return None, 'No response'
        response = str(response)
        if _is_blank(response):
            return None, 'Blank grid'
        if not response.isdigit():
            return None, f'Unrecognised: {response}'
        # Name the category when it is known, but still draw the real grid.
        label = row.get('response_type')
        if label is None or pd.isna(label):
            return response, 'grid'
        return response, _format_concept(label)

    # -- Discrimination: the response names one of the offered matrices -------
    choice = row.get('choice')
    if choice is not None and not pd.isna(choice) and choice in CHOICE_TO_MATRIX:
        name = CHOICE_TO_MATRIX[choice]
        return matrix_target(item, name, matrix_level), _format_concept(choice)

    if is_missing:
        return None, 'No response'

    response = str(response)

    # A category label stored directly (e.g. 'D_Concept').
    if response.startswith('D_') or response == 'C':
        return matrix_target(item, response, matrix_level), _format_concept(response)

    return None, f'Unrecognised: {response}'


def _subject_col(df: pd.DataFrame) -> Optional[str]:
    for col in SUBJECT_COLS:
        if col in df.columns:
            return col
    return None


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def resolve_task(row: pd.Series, items_source: Any = None, task: str = None) -> str:
    """
    Determine the task for a row.

    `HumanEval.df` carries a `task` column; `Eval.df` does not, because an
    `Eval` holds one task per file. Fall back to the dataset, then to an
    explicit argument.
    """
    if task is not None:
        return str(task).lower()
    if 'task' in row.index and not pd.isna(row['task']):
        return str(row['task']).lower()
    if items_source is not None and hasattr(items_source, 'task'):
        return str(items_source.task).lower()
    raise ValueError(
        'Could not determine the task. `Eval.df` has no `task` column, so '
        'pass an AmbigousARCDataset as `items`, or set task=... explicitly.'
    )


def plot_response(
    row: pd.Series,
    items: Any,
    xdim: int = None,
    task: str = None,
    return_fig=False,
):
    """
    Plot one response.

    Parameters
    ----------
    row : pandas.Series
        A row from `Eval(...).df` or `HumanEval(...).df`.
    items : str | list[dict] | AmbigousARCDataset
        Item source, see `items_to_dict`.
    task : str, optional
        Needed only when the frame has no `task` column and `items` is not
        a dataset.
    """
    items_source = items
    items = items if isinstance(items, dict) else items_to_dict(items)

    item_id = str(row.get('item_id', row.get('itemid')))
    item = get_item(items, item_id)
    task = resolve_task(row, items_source, task)
    matrix_level = row.get('matrix_level')

    xdim = xdim if xdim is not None else item.get('xdim')
    ydim = item.get('ydim', xdim)

    subject = None
    for col in SUBJECT_COLS:
        if col in row.index:
            subject = row[col]
            break

    # Title carries the context needed to interpret a surprising response.
    bits = [f'{item_id}', f'{task.capitalize()}', f'{_format_concept(item.get("concept"))}']
    if subject is not None:
        bits.insert(0, f'{subject}')
    if matrix_level is not None and not pd.isna(matrix_level):
        bits.append(f'Level: {matrix_level}')
    if 'rt' in row.index and not pd.isna(row['rt']):
        bits.append(f'RT: {row["rt"]} ms')
    title = '  |  '.join(str(b) for b in bits)

    if task == 'recognition':
        fig = _plot_recognition(row, item, xdim, ydim, title)
    elif task in {'generation', 'discrimination'}:
        fig = _plot_matrix_task(row, item, xdim, ydim, matrix_level, title, task)
    else:
        raise ValueError(
            f"Unknown task '{task}'. Expected generation, discrimination "
            f'or recognition.'
        )

    if return_fig:
        plt.close(fig)
        return fig
    # Return nothing once shown: Jupyter would otherwise render the returned
    # figure a second time. Matches `plot_utils.plot_item`.
    plt.show()


def _plot_matrix_task(row, item, xdim, ydim, matrix_level, title, task):
    response_matrix, status = resolve_response(row, item, task)

    fig = plt.figure(figsize=(6, 5.7))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1, 0.35, 1], height_ratios=[1, 1, 1.15], hspace=0.55
    )

    # A -> B
    _draw_grid(fig.add_subplot(gs[0, 0]), item.get('A'), 'A', xdim, ydim)
    _draw_arrow(fig.add_subplot(gs[0, 1]))
    _draw_grid(fig.add_subplot(gs[0, 2]), item.get('B'), 'B', xdim, ydim)

    # C -> response
    _draw_grid(fig.add_subplot(gs[1, 0]), item.get('C'), 'C', xdim, ydim)
    _draw_arrow(fig.add_subplot(gs[1, 1]))
    _draw_grid(fig.add_subplot(gs[1, 2]), response_matrix, f'Response: {status}', xdim, ydim)

    # Candidate answers
    options_gs = gs[2, :].subgridspec(1, 3, wspace=0.45)
    for idx, name in enumerate(OPTION_MATRICES):
        label = name.replace('_', ' ')
        if name == 'D_Matrix' and str(matrix_level).lower() == 'row':
            label += ' (row)'
        _draw_grid(
            fig.add_subplot(options_gs[0, idx]),
            matrix_target(item, name, matrix_level),
            label, xdim, ydim,
        )

    fig.suptitle(title, fontsize=13, fontweight='bold')
    return fig


def _plot_recognition(row, item, xdim, ydim, title):
    correct = item.get('concept')

    # `Eval` resolves the letter to a concept in `choice`; `HumanEval` stores
    # the concept in `response`.
    selected = row.get('choice')
    if selected is None or pd.isna(selected):
        selected = row.get('response')
    if isinstance(selected, float) and pd.isna(selected):
        selected = None

    fig = plt.figure(figsize=(6, 5.7))
    gs = fig.add_gridspec(
        3, 3, width_ratios=[1, 0.35, 1], height_ratios=[1, 1, 1.25], hspace=0.55
    )

    _draw_grid(fig.add_subplot(gs[0, 0]), item.get('A'), 'A', xdim, ydim)
    _draw_arrow(fig.add_subplot(gs[0, 1]))
    _draw_grid(fig.add_subplot(gs[0, 2]), item.get('B'), 'B', xdim, ydim)

    _draw_grid(fig.add_subplot(gs[1, 0]), item.get('C'), 'C', xdim, ydim)
    _draw_grid(fig.add_subplot(gs[1, 1:]), item.get('D_Concept'), 'D Concept', xdim, ydim)

    options_ax = fig.add_subplot(gs[2, :])
    _draw_concept_options(
        options_ax,
        concept_options=item.get('concept_options') or [],
        selected=selected,
        correct=correct,
    )

    colour = 'dimgray' if selected is None else (
        'forestgreen' if selected == correct else 'firebrick'
    )
    fig.suptitle(
        f'{title}\nSelected: {_format_concept(selected)}',
        fontsize=13, fontweight='bold', color=colour,
    )
    return fig


# -----------------------------------------------------------------------------
# Browser
# -----------------------------------------------------------------------------

class ResponseBrowser:
    """
    Step through the responses to one item.

    Works on either `Eval(...).df` or `HumanEval(...).df`.

    Parameters
    ----------
    df : pandas.DataFrame
        Response frame.
    items : str | list[dict] | AmbigousARCDataset
        Item source.
    item_id : str, optional
        Either a versioned item id ('02ue5B_2') or a base id ('02ue5B').
        A base id pools every mirror of that item. Give this or `item_index`.
    item_index : int, optional
        Position within the item source.
    subject : optional
        Restrict to one participant or model.
    task : str, optional
        Restrict to one task. Only useful for frames holding several.
    main_id : bool, optional
        Override the automatic detection above. `True` forces pooling of
        mirrors even when a versioned id was given; `False` forces an exact
        match. Leave as None to detect from `item_id`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        items: Any,
        item_id: str = None,
        item_index: int = None,
        subject: Any = None,
        task: str = None,
        main_id: bool = None,
    ):
        if (item_id is None) == (item_index is None):
            raise ValueError('Provide exactly one of `item_id` or `item_index`.')

        self.items = items if isinstance(items, dict) else items_to_dict(items)
        self.df = df
        self.subject_col = _subject_col(df)

        known_item_ids = set(self.items)
        known_main_ids = {
            rec.get('main_id') for rec in self.items.values()
            if rec.get('main_id')
        }

        if item_index is not None:
            keys = list(self.items)
            if not 0 <= item_index < len(keys):
                raise IndexError(
                    f'`item_index` must be between 0 and {len(keys) - 1}.'
                )
            self.item_id = keys[item_index]
            detected_main = False
        else:
            self.item_id = str(item_id)
            if self.item_id in known_item_ids:
                detected_main = False
            elif self.item_id in known_main_ids:
                detected_main = True
            else:
                raise KeyError(
                    f"'{self.item_id}' is not an item id or a base id in the "
                    f'item source ({len(known_item_ids)} items, '
                    f'{len(known_main_ids)} base ids).'
                )

        # An explicit argument wins over detection.
        self.main_id = detected_main if main_id is None else bool(main_id)

        if self.main_id:
            target = self.item_id.rsplit('_', 1)[0]
            id_col = 'item_main_id' if 'item_main_id' in df.columns else None
            if id_col is None:
                # Fall back to deriving the base id from the versioned one.
                raw_col = 'item_id' if 'item_id' in df.columns else 'itemid'
                ids = df[raw_col].astype(str).str.rsplit('_', n=1).str[0]
                matches = df[ids == target]
            else:
                matches = df[df[id_col].astype(str) == target]
        else:
            id_col = 'item_id' if 'item_id' in df.columns else 'itemid'
            target = self.item_id
            matches = df[df[id_col].astype(str) == target]

        if subject is not None:
            if self.subject_col is None:
                raise KeyError(
                    f'No respondent column found. Looked for {SUBJECT_COLS}.'
                )
            matches = matches[matches[self.subject_col] == subject]
        if task is not None:
            matches = matches[matches['task'].astype(str).str.lower() == task.lower()]

        if matches.empty:
            raise ValueError(
                f"No responses for {'base id' if self.main_id else 'item'} "
                f"'{target}'"
                + (f", subject '{subject}'" if subject is not None else '')
                + (f", task '{task}'" if task is not None else '')
                + '.'
            )

        # Browse mirrors in order, then respondents, so stepping through a
        # pooled base id is not arbitrary.
        sort_cols = [c for c in ['item_id', 'itemid', self.subject_col]
                     if c and c in matches.columns]
        if sort_cols:
            matches = matches.sort_values(sort_cols, kind='stable')

        self.matches = matches.reset_index(drop=True)
        self.index = 0
        self.task = task
        self.items_source = items

    @property
    def n_responses(self) -> int:
        return len(self.matches)

    @property
    def current_row(self) -> pd.Series:
        return self.matches.iloc[self.index]

    def show(self, return_fig=False):
        row = self.current_row
        label = f'Response {self.index + 1} of {self.n_responses}'
        if self.main_id:
            label += f"  (mirror {row.get('item_id', row.get('itemid'))})"
        print(label)
        return plot_response(
            row, self.items_source, task=self.task, return_fig=return_fig
        )

    def next(self, steps: int = 1):
        self.index = (self.index + steps) % self.n_responses
        return self.show()

    def previous(self, steps: int = 1):
        self.index = (self.index - steps) % self.n_responses
        return self.show()

    def go_to(self, index: int):
        self.index = index % self.n_responses
        return self.show()

    def summary(self) -> pd.DataFrame:
        cols = [
            self.subject_col, 'item_id', 'task', 'response', 'choice',
            'response_type', 'matrix_level', 'rt', 'kidsarc_response_id',
        ]
        cols = [c for c in cols if c and c in self.matches.columns]
        return self.matches[cols].copy()
