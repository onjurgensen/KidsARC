"""
Evaluation of human responses, mirroring `utils/eval.py`.

`HumanEval` is to the human data what `Eval` is to a model results file:
one object per task, exposing a tidy `.df` with one row per response.

    from utils.human_eval import HumanEval

    generation     = HumanEval('path/to/responses.csv', task='generation')
    discrimination = HumanEval('path/to/responses.csv', task='discrimination')
    recognition    = HumanEval('path/to/responses.csv', task='recognition')

    generation.print()
    generation.df.groupby('concept')['concept_response'].mean()

Expected response columns: `kidsarc_response_id`, `participant_fk`, `itemid`,
`response`, `task`. Optional and carried through to `.df` when present:
`rt`, `matrix_level`, `correct`, `saveitem_timestamp`. Any other columns are
ignored, so a wider export is fine.

The response-type columns match `Eval.df` so the two can be concatenated:
`concept_response`, `matrix_response`, `duplicate_response`, `other_response`.
Two columns are added that `Eval` does not have -- `no_response` and
`response_type` -- see the notes at the bottom of this file.
"""

import os
import json
import warnings
from typing import *

import numpy as np
import pandas as pd
from rich import print as rprint

# Paths are resolved relative to this file, not the working directory.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_DIR = os.path.join(_BASE_DIR, '..', 'data', 'human_data_collection')
HUMAN_ITEMS_FILE = os.path.join(HUMAN_DIR, 'AmbigousARC.json')

TASKS = ['generation', 'discrimination', 'recognition']

# Discrimination responses are stored as labels rather than grids.
DISCRIMINATION_LABELS = {
    'D_Concept': 'concept',
    'D_Matrix': 'matrix',
    'D_Duplicate': 'duplicate',
    'D_Random': 'other',
}


def is_missing(response: Any) -> bool:
    """True if the response is NULL."""
    return response is None or (isinstance(response, float) and np.isnan(response))


def is_blank_grid(response: Any) -> bool:
    """
    True if the response is an all-zero grid.

    Zero is the background colour, so an all-zero grid is a submitted-but-empty
    canvas. `human_preprocessing.item_left_unanswered` treats these as
    unanswered and so do we.
    """
    if is_missing(response):
        return False
    s = str(response)
    return s.isdigit() and set(s) == {'0'}


def clean_response(response: Any) -> Optional[str]:
    """
    Return the response as a canonical string, or None if it is not a
    usable answer.

    The human interface stores grids already flattened and unbracketed, so
    unlike `eval.filter_response` there is nothing to parse out of free text.
    """
    if is_missing(response) or is_blank_grid(response):
        return None
    return str(response)


def get_matrix_target(item: Dict, matrix_level: Any = None) -> str:
    """
    The literal ("matrix") answer for an item.

    Items shown in the row condition have their own literal answer; everything
    else uses the pixel-level one. See `prompt_utils.get_d_matrix`.
    """
    if str(matrix_level).lower() == 'row':
        return item['D_Matrix_Row']
    return item['D_Matrix']


class ParticipantEval:
    """
    One participant's responses for one task.

    The analogue of `ModelEval`: classifies each response and exposes the
    per-category proportions.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: Any,
        task: str,
        items: Dict[str, Dict],
    ):
        self.data = data.reset_index(drop=True)
        self.name = name
        self.task = task
        self.items = items

        self.item_ids = self.data['itemid'].astype(str).tolist()
        self.item_records = [self.items[i] for i in self.item_ids]
        self.concepts = [item['concept'] for item in self.item_records]
        self.size = len(self.data)

        self.clean_responses = [clean_response(r) for r in self.data['response']]
        self.no_responses = [1 if r is None else 0 for r in self.clean_responses]
        self.valid_responses_n = self.size - sum(self.no_responses)

        if self.task == 'generation':
            self._classify_generation()
        elif self.task == 'discrimination':
            self._classify_discrimination()
        elif self.task == 'recognition':
            self._classify_recognition()
        else:
            raise ValueError(f'Invalid task "{self.task}"')

        self.response_types = self._get_response_types()
        # Generation has no answer choice -- `Eval` omits the column for that
        # task too. `response_type` carries the classification instead.

        # Proportions are over valid responses only, so that a participant who
        # skipped half the items is not recorded as giving "other" answers.
        self.concept_prop = self._prop(self.concept_responses)
        self.other_prop = self._prop(self.other_responses)
        if self.task != 'recognition':
            self.matrix_prop = self._prop(self.matrix_responses)
            self.duplicate_prop = self._prop(self.duplicate_responses)
        self.no_response_prop = np.mean(self.no_responses) if self.size else np.nan

    # -- classification ----------------------------------------------------

    def _classify_generation(self):
        """Exact-match the submitted grid against each candidate answer."""
        matrix_levels = self.data.get('matrix_level', pd.Series([None] * self.size))

        self.concept_responses = []
        self.matrix_responses = []
        self.matrix_pixel_responses = []
        self.matrix_row_responses = []
        self.duplicate_responses = []

        for response, item, level in zip(
            self.clean_responses, self.item_records, matrix_levels
        ):
            if response is None:
                for lst in (
                    self.concept_responses, self.matrix_responses,
                    self.matrix_pixel_responses, self.matrix_row_responses,
                    self.duplicate_responses,
                ):
                    lst.append(0)
                continue

            self.concept_responses.append(int(response == item['D_Concept']))
            self.matrix_responses.append(int(response == get_matrix_target(item, level)))
            # Generation responses are free-form grids, so they can be scored
            # against both literal rules regardless of which was assigned.
            self.matrix_pixel_responses.append(int(response == item['D_Matrix']))
            self.matrix_row_responses.append(int(response == item['D_Matrix_Row']))
            self.duplicate_responses.append(
                int(response in [item['A'], item['B'], item['C']])
            )

        self.other_responses = self._residual(
            self.concept_responses, self.matrix_responses, self.duplicate_responses
        )
        # Generation has no explicit answer choice; `response_type` carries it.
        self.choices = None

    def _classify_discrimination(self):
        """
        The stored response is already the chosen option's label, so the
        classification is a lookup rather than a comparison.

        Note that the label is `D_Matrix` whether the participant saw the
        pixel or the row variant; `matrix_level` records which.
        """
        self.choices = [
            DISCRIMINATION_LABELS.get(r) if r is not None else None
            for r in self.clean_responses
        ]
        unknown = {
            r for r, c in zip(self.clean_responses, self.choices)
            if r is not None and c is None
        }
        if unknown:
            raise ValueError(f'Unknown discrimination response(s): {sorted(unknown)}')

        self.concept_responses = [int(c == 'concept') for c in self.choices]
        self.matrix_responses = [int(c == 'matrix') for c in self.choices]
        self.duplicate_responses = [int(c == 'duplicate') for c in self.choices]
        self.other_responses = [int(c == 'other') for c in self.choices]

        # The stored label is `D_Matrix` in both conditions, so which literal
        # rule the participant endorsed is given by `matrix_level`.
        levels = self.data.get('matrix_level', pd.Series([None] * self.size))
        levels = [str(l).lower() for l in levels]
        self.matrix_pixel_responses = [
            int(m and l == 'pixel') for m, l in zip(self.matrix_responses, levels)
        ]
        self.matrix_row_responses = [
            int(m and l == 'row') for m, l in zip(self.matrix_responses, levels)
        ]

    def _classify_recognition(self):
        """Compare the chosen concept label with the item's concept."""
        self.choices = list(self.clean_responses)
        self.concept_responses = [
            int(r == item['concept']) if r is not None else 0
            for r, item in zip(self.clean_responses, self.item_records)
        ]
        self.other_responses = self._residual(self.concept_responses)

    # -- helpers -----------------------------------------------------------

    def _residual(self, *response_lists) -> List[int]:
        """
        1 where a valid response matched none of the named categories.

        Unlike `eval.get_all_zero_indices`, a missing response is *not*
        counted as "other".
        """
        return [
            0 if no_response else int(all(x == 0 for x in items))
            for no_response, *items in zip(self.no_responses, *response_lists)
        ]

    def _get_response_types(self) -> List[str]:
        types = []
        for i in range(self.size):
            if self.no_responses[i]:
                types.append('no_response')
            elif self.concept_responses[i]:
                types.append('concept')
            elif self.task != 'recognition' and self.matrix_responses[i]:
                types.append('matrix')
            elif self.task != 'recognition' and self.duplicate_responses[i]:
                types.append('duplicate')
            else:
                types.append('other')
        return types

    def _prop(self, responses: List[int]) -> float:
        if not self.valid_responses_n:
            return np.nan
        valid = [r for r, missing in zip(responses, self.no_responses) if not missing]
        return float(np.mean(valid))


class HumanEval:
    """
    Human responses for one task, mirroring `Eval`.

    Parameters
    ----------
    data : str | pandas.DataFrame
        Path to the response CSV, or an already-loaded frame. Required.
    task : str
        One of 'generation', 'discrimination', 'recognition'.
    items : str | list[dict]
        Path to the items JSON, or the loaded list. Defaults to the human
        collection's `AmbigousARC.json`.
    no_response_thresh : float, optional
        Drop participants whose proportion of valid responses falls below
        this. Mirrors `Eval`'s argument of the same name. Off by default.
    participants : list, optional
        Restrict to these participant IDs.
    """

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        task: str = 'generation',
        items: Union[str, List[Dict]] = HUMAN_ITEMS_FILE,
        no_response_thresh: float = None,
        participants: List = None,
        participant_col: str = 'participant_fk',
    ):
        if task not in TASKS:
            raise ValueError(f'Invalid task "{task}". Choose from {TASKS}.')

        self.task = task
        self.participant_col = participant_col

        # Items, keyed by versioned item id
        if isinstance(items, str):
            items = json.load(open(items, 'rb'))
        self.items = {str(item['item_id']): item for item in items}

        # Responses
        if isinstance(data, str):
            data = pd.read_csv(data)
        data = data.copy()
        data['task'] = data['task'].astype(str).str.lower()
        data = data[data['task'] == self.task]

        if participants is not None:
            data = data[data[participant_col].isin(participants)]

        # The raw export contains rows whose item id is NULL or unknown (test
        # sessions, aborted trials). Drop them, but say so rather than
        # silently shrinking the dataset.
        resolvable = data['itemid'].astype(str).isin(self.items)
        self.dropped_unresolved = data[~resolvable].copy()
        if len(self.dropped_unresolved):
            unknown = sorted(set(self.dropped_unresolved['itemid'].astype(str)))
            warnings.warn(
                f'Dropped {len(self.dropped_unresolved)} response(s) whose item id '
                f'is not in the items file: {unknown[:5]}'
            )
        data = data[resolvable]

        self.data = data.reset_index(drop=True)
        self.all_participants_n = self.data[participant_col].nunique()

        # One ParticipantEval per participant
        self.participants = [
            ParticipantEval(group, name, self.task, self.items)
            for name, group in self.data.groupby(participant_col, sort=True)
        ]

        # Optional exclusion, mirroring Eval
        self.excluded_no_response = []
        if no_response_thresh:
            self.excluded_no_response = [
                p for p in self.participants
                if p.valid_responses_n < no_response_thresh * p.size
            ]
            self.participants = [
                p for p in self.participants if p not in self.excluded_no_response
            ]

        self.participant_names = [p.name for p in self.participants]
        self.df = self.to_pd()

    def to_pd(self) -> pd.DataFrame:
        if not self.participants:
            print('No participants to evaluate')
            return None

        frames = []
        for p in self.participants:
            cfg = {
                'item_main_id': [i.rsplit('_', 1)[0] for i in p.item_ids],
                'item_id': p.item_ids,
                'mirror': [i.rsplit('_', 1)[1] for i in p.item_ids],
                'participant': p.name,
                'task': self.task,
                'concept': p.concepts,
                'response': p.clean_responses,
                'response_type': p.response_types,
                'concept_response': p.concept_responses,
                'other_response': p.other_responses,
                'no_response': p.no_responses,
            }
            if self.task != 'recognition':
                cfg['matrix_response'] = p.matrix_responses
                cfg['matrix_pixel_response'] = p.matrix_pixel_responses
                cfg['matrix_row_response'] = p.matrix_row_responses
                cfg['duplicate_response'] = p.duplicate_responses

            if p.choices is not None:
                cfg['choice'] = p.choices

            frame = pd.DataFrame(cfg)

            # Carry through columns that are useful for filtering downstream.
            # Anything present in the source is kept; nothing is required.
            for col in [
                'rt', 'matrix_level', 'kidsarc_response_id',
                'correct', 'saveitem_timestamp',
            ]:
                if col in p.data.columns:
                    frame[col] = p.data[col].values

            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    def __str__(self):
        s = f'Task: {self.task.capitalize()}\n'
        s += f'Number of participants: {len(self.participants)}\n'
        s += f'Number of responses: {len(self.df)}\n\n'
        s += f'Concept: {self.df["concept_response"].mean():.2f}\n'
        if self.task != 'recognition':
            s += f'Matrix: {self.df["matrix_response"].mean():.2f}\n'
            s += f'Duplicate: {self.df["duplicate_response"].mean():.2f}\n'
        s += f'Other: {self.df["other_response"].mean():.2f}\n'
        s += f'No response: {self.df["no_response"].mean():.2f}'
        if self.excluded_no_response:
            s += (
                f'\n\n{len(self.excluded_no_response)}/{self.all_participants_n} '
                f'participants excluded due to high no-response rate.'
            )
        return s

    def print(self):
        rprint(str(self))


# -----------------------------------------------------------------------------
# Differences from utils/eval.py, deliberate
#
# 1. `no_response` is an explicit column, and proportions are computed over
#    valid responses only. `ModelEval` folds missing and unparseable responses
#    into `other_response`. Human data contains both NULL responses and
#    submitted-but-empty (all-zero) grids, and counting either as an "other"
#    answer would inflate that category. Check `df['no_response'].mean()` on
#    your own file to see how much this matters.
#
# 2. `other_response` is the residual in every task. In `Eval` it is the
#    residual for generation but specifically `D_Random` for discrimination.
#    Here `choice == 'other'` still identifies D_Random in discrimination.
#
# 3. The row-level literal answer is used when `matrix_level == 'row'`, and
#    `matrix_pixel_response` / `matrix_row_response` are reported separately.
#    `human_preprocessing.score_discrimination` compares against the string
#    'D_Matrix_Row', which does not occur in the response data, so its
#    row-literal score is always zero.
#
# 4. No robustness score. `ModelEval.calculate_robustness` needs each
#    respondent to see several mirrors of the same item. Whether that holds
#    depends on how items were assigned in your data collection -- see
#    `mirror_coverage()` below. If participants routinely see multiple
#    mirrors, porting `calculate_robustness` is worth doing.
# -----------------------------------------------------------------------------


def mirror_coverage(data: Union[str, pd.DataFrame]) -> pd.Series:
    """
    How many mirrors of the same main item each participant saw, per task.

    Robustness (see note 4 above) is only meaningful if this is regularly
    greater than 1. Returns the distribution of counts.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
    data = data.copy()
    data['main_id'] = data['itemid'].astype(str).str.rsplit('_', n=1).str[0]
    counts = data.groupby(['participant_fk', 'task', 'main_id']).size()
    return counts.value_counts().sort_index()