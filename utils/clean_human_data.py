"""
Clean Kids ARC human data for one task.

    from utils.clean_human_data import load_data, clean_human_data

    users, responses, process = load_data(USERS_CSV, RESPONSES_CSV, PROCESS_CSV)
    resp, proc, summary = clean_human_data(
        users, responses, process, task='generation', min_rt_ms=1000
    )

Or from the repo root, using the paths hardcoded below:

    python -m utils.clean_human_data

or overriding them:

    python -m utils.clean_human_data users.csv responses.csv process.csv --out data/clean

Trials are excluded in this order, and each step is logged:
    1. other tasks
    2. participants not in the registered-users file (`participant_id_arc`)
    3. items whose base id (`main_id`) is not in items_updated.json
    4. item variants with no known solution
    5. blank responses (missing, or an all-zero grid)
    6. rt missing or below `min_rt_ms`

Responses are scored against the exact variant shown (`itemid`, e.g. a
mirror `_3`); `main_id` is the shared item reference in all outputs.

Process rows are kept only for (participant, item, task) trials that
survive the response cleaning.
"""

import os
import json
import argparse
import warnings
from typing import *

import pandas as pd

from utils.human_eval import is_blank_grid, is_missing

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_BASE_DIR, '..'))

USERS_FILE = 'data/human_data_collection/user_data_merged.csv'
RESPONSES_FILE = 'data/imports/kidsarc.csv'
PROCESS_FILE = 'data/imports/kidsarc_process.csv'
OUT_DIR = 'data/human_data_collection'

ITEMS_DIR = os.path.join(_BASE_DIR, '..', 'data', 'items')
ITEMS_FILE = os.path.join(ITEMS_DIR, 'items_updated.json')  # defines the valid item set
SOLUTION_FILES = [
    os.path.join(ITEMS_DIR, 'items_updated_exp.json'),
    os.path.join(ITEMS_DIR, 'items_original_exp.json'),
]
SOLUTION_COLS = ['item_id', 'main_id', 'A', 'B', 'C', 'D_Concept', 'D_Matrix', 'concept']
DUPLICATE_TERMS = ['B', 'C']

TRIAL_KEY = ['participant_fk', 'itemid', 'task']
USER_COLS = ['participant_id', 'participant_id_arc', 'research_study', 'age', 'language']


# -- loading ----------------------------------------------------------------

def load_data(users_path: str, responses_path: str, process_path: str):
    """Read the three CSVs with explicit types (grids stay strings, NULL -> NaN)."""
    read = dict(na_values=['NULL'])
    users = pd.read_csv(users_path, **read)
    responses = pd.read_csv(
        responses_path, dtype={'itemid': str, 'response': str, 'task': str}, **read
    )
    process = pd.read_csv(
        process_path,
        dtype={'itemid': str, 'state': str, 'task': str, 'page_id': str,
               'cell': 'Int64', 'color': 'Int64'}, **read
    )

    # Same integer type on both sides of every join
    users['participant_id_arc'] = pd.to_numeric(users['participant_id_arc']).astype('Int64')
    for df in (responses, process):
        df['participant_fk'] = pd.to_numeric(df['participant_fk']).astype('Int64')
        df['task'] = df['task'].str.lower()
        # Humans see one variant per item, so items are identified by base id
        df['main_id'] = df['itemid'].str.rsplit('_', n=1).str[0]
    responses['rt'] = pd.to_numeric(responses['rt'])

    return users, responses, process


# -- cleaning ---------------------------------------------------------------

def _log(label: str, df: pd.DataFrame, before: pd.DataFrame = None):
    n, p = len(df), df['participant_fk'].nunique()
    if before is None:
        print(f'{label:<34} {n:>7} trials {p:>5} participants')
    else:
        dn, dp = len(before) - n, before['participant_fk'].nunique() - p
        print(f'{"- " + label:<34} {-dn:>7} trials {-dp:>5} participants'
              f'   -> {n} / {p}')


def clean_responses(
    responses: pd.DataFrame,
    users: pd.DataFrame,
    valid_ids: Set[str],
    solutions: pd.DataFrame,
    task: str = 'generation',
    min_rt_ms: float = 1000,
) -> pd.DataFrame:
    registered = set(users['participant_id_arc'].dropna())
    known_variants = set(solutions['item_id'])

    # Each filter returns True for rows to DROP
    filters = [
        (f'task != {task!r}',          lambda d: d['task'] != task),
        ('unregistered participant',   lambda d: ~d['participant_fk'].isin(registered)),
        ('item not in items_updated',  lambda d: ~d['main_id'].isin(valid_ids)),
        ('no solution for variant',    lambda d: ~d['itemid'].isin(known_variants)),
        ('blank response',             lambda d: d['response'].apply(
                                           lambda r: is_missing(r) or is_blank_grid(r))),
        (f'rt missing or < {min_rt_ms} ms', lambda d: ~(d['rt'] >= min_rt_ms)),
    ]

    df = responses.copy()
    _log('raw', df)
    for label, drop in filters:
        before = df
        df = df[~drop(df)]
        _log(label, df, before)

    dupes = df.duplicated(TRIAL_KEY, keep=False)
    if dupes.any():
        warnings.warn(
            f'{dupes.sum()} cleaned trials share a (participant, item, task) key; '
            'their process rows cannot be told apart.'
        )
    return df.reset_index(drop=True)


def clean_process(process: pd.DataFrame, clean_resp: pd.DataFrame) -> pd.DataFrame:
    """Keep process rows belonging to a trial that survived response cleaning."""
    keys = clean_resp[TRIAL_KEY].drop_duplicates()
    proc = process.merge(keys, on=TRIAL_KEY, how='inner')

    print(f'\nprocess: kept {proc["page_id"].nunique()} of '
          f'{process["page_id"].nunique()} trials (page_id), {len(proc)} of {len(process)} rows')

    # Cleaned trials with no process data at all (e.g. sessions from before
    # process logging existed)
    has_proc = keys.merge(proc[TRIAL_KEY].drop_duplicates(), on=TRIAL_KEY, how='left',
                          indicator=True)['_merge'].eq('both')
    if (~has_proc).any():
        print(f'process: {(~has_proc).sum()} cleaned trials have no process rows')

    # Sanity check: final submitted state should equal the stored response
    submits = (proc[proc['action'] == 'submit']
               .sort_values('event_idx')
               .drop_duplicates(TRIAL_KEY, keep='last'))
    check = submits.merge(clean_resp[TRIAL_KEY + ['response']], on=TRIAL_KEY)
    mismatch = (check['state'] != check['response']).sum()
    if mismatch:
        warnings.warn(f'{mismatch} trials: last submitted state != stored response')

    return proc.reset_index(drop=True)


# -- scoring and summary ----------------------------------------------------

def load_items(
    items_path: str = ITEMS_FILE,
    solution_paths: List[str] = SOLUTION_FILES,
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Returns the valid base ids and a solutions table with one row per variant.

    Warns about items whose _1 grids differ between solution files: their
    other variants are mirrors of the earlier version.
    """
    valid_ids = {str(item['id']) for item in json.load(open(items_path, 'rb'))}

    frames = []
    for rank, path in enumerate(solution_paths):
        df = pd.DataFrame(json.load(open(path, 'rb')))
        frames.append(df.reindex(columns=SOLUTION_COLS).assign(_rank=rank))
    all_sol = pd.concat(frames, ignore_index=True)

    grid_cols = ['A', 'B', 'C', 'D_Concept', 'D_Matrix']
    conflicting = (all_sol.drop_duplicates(['item_id'] + grid_cols)
                   .loc[lambda d: d['item_id'].duplicated(keep=False), 'main_id'])
    modified = sorted(set(conflicting) & valid_ids)
    if modified:
        warnings.warn(f'Items changed between solution files: {modified}. Their '
                      'non-overridden variants are mirrors of the earlier version.')

    solutions = (all_sol.sort_values('_rank')
                 .drop_duplicates('item_id')
                 .drop(columns='_rank')
                 .reset_index(drop=True))
    return valid_ids, solutions


def score_responses(clean_resp: pd.DataFrame, solutions: pd.DataFrame) -> pd.DataFrame:
    """
    Exact-match each grid against the solutions for the variant shown, as in
    `HumanEval._classify_generation`. Blank trials are already removed.
    """
    df = clean_resp.merge(solutions.drop(columns='main_id'),
                          left_on='itemid', right_on='item_id', how='left')
    hits = ['concept_response', 'matrix_response', 'duplicate_response']
    copied = df[['A', 'B', 'C']].eq(df['response'], axis=0)
    df['concept_response'] = (df['response'] == df['D_Concept']).astype(int)
    df['matrix_response'] = (df['response'] == df['D_Matrix']).astype(int)
    df['duplicate_response'] = copied[DUPLICATE_TERMS].any(axis=1).astype(int)
    df['duplicate_of'] = copied.apply(lambda row: ';'.join(row.index[row]), axis=1).replace('', pd.NA)
    df['other_response'] = (df[hits].sum(axis=1) == 0).astype(int)
    df['response_type'] = (df[hits].idxmax(axis=1).str.replace('_response', '')
                           .where(df['other_response'] == 0, 'other'))
    return df[list(clean_resp.columns) + ['concept'] + hits
              + ['other_response', 'response_type', 'duplicate_of']]


def summarise(scored: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: demographics, accuracy, and items answered."""
    agg = {
        'n_trials': ('itemid', 'size'),
        'concept_acc': ('concept_response', 'mean'),
        'matrix_acc': ('matrix_response', 'mean'),
        'duplicate_prop': ('duplicate_response', 'mean'),
        'items': ('main_id', list),
    }

    per_participant = scored.groupby('participant_fk').agg(**agg).reset_index()
    summary = users[USER_COLS].merge(
        per_participant, left_on='participant_id_arc', right_on='participant_fk',
        how='inner',
    ).drop(columns='participant_fk')

    n_none = users['participant_id_arc'].nunique() - len(summary)
    print(f'\nsummary: {len(summary)} participants; '
          f'{n_none} registered participants have no valid trials')
    return summary


# -- entry point ------------------------------------------------------------

def clean_human_data(
    users: pd.DataFrame,
    responses: pd.DataFrame,
    process: pd.DataFrame,
    task: str = 'generation',
    min_rt_ms: float = 1000,
    items: str = ITEMS_FILE,
    solution_files: List[str] = SOLUTION_FILES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if task != 'generation':
        raise NotImplementedError('Scoring is implemented for the generation task only')
    valid_ids, solutions = load_items(items, solution_files)

    if users['participant_id_arc'].duplicated().any():
        warnings.warn('participant_id_arc is not unique in the users file')

    resp = clean_responses(responses, users, valid_ids, solutions, task, min_rt_ms)
    proc = clean_process(process, resp)
    resp = score_responses(resp, solutions)
    summary = summarise(resp, users)
    return resp, proc, summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    repo = lambda path: os.path.join(_REPO_DIR, path)
    ap.add_argument('users', nargs='?', default=repo(USERS_FILE))
    ap.add_argument('responses', nargs='?', default=repo(RESPONSES_FILE))
    ap.add_argument('process', nargs='?', default=repo(PROCESS_FILE))
    ap.add_argument('--task', default='generation')
    ap.add_argument('--min-rt', type=float, default=1000, help='ms')
    ap.add_argument('--items', default=ITEMS_FILE)
    ap.add_argument('--solution-files', nargs='+', default=SOLUTION_FILES)
    ap.add_argument('--out', default=repo(OUT_DIR))
    args = ap.parse_args()

    resp, proc, summary = clean_human_data(
        *load_data(args.users, args.responses, args.process),
        task=args.task, min_rt_ms=args.min_rt,
        items=args.items, solution_files=args.solution_files,
    )
    os.makedirs(args.out, exist_ok=True)
    resp.to_csv(os.path.join(args.out, f'kidsarc_{args.task}.csv'), index=False)
    proc.to_csv(os.path.join(args.out, f'process_{args.task}.csv'), index=False)
    summary.assign(items=summary['items'].str.join(';')).to_csv(
        os.path.join(args.out, f'{args.task}_summary.csv'), index=False
    )