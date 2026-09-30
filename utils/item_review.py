"""
Review newly authored items before adding them to the item set.

Workflow:

    from utils.item_review import ItemReviewer, concept_table

    reviewer = ItemReviewer(new_items_df, items_data)
    reviewer.report()          # what the CSV looks like before you start

    reviewer.show()            # draw the current item
    reviewer.accept()          # keep it, with the concept from the CSV
    reviewer.accept('my_new_concept')   # refused: concept not in the data
    reviewer.accept('my_new_concept', new_concept=True)   # confirmed
    reviewer.reject()          # skip it
    reviewer.next() / .previous() / .go_to(i)

    reviewer.summary()         # items per concept by grid size
    reviewer.items             # the updated list -- a copy, the original is untouched
    reviewer.save('data/new_items/items_updated.json')

`accept` and `reject` advance automatically, so a review pass is just
show / accept / show / accept.

Duplicates are matched on grid content, not on `id`, since ids from an
external authoring tool are not comparable to the existing ones.
"""

import json
import copy
from typing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot_utils import plot_matrix
from utils.prompt_utils import convert_to_array, get_d_matrix

# Grids that identify an item. Two items matching on any of these are the
# same item as far as review is concerned.
GRID_FIELDS = ['A', 'B', 'C', 'D_Concept']

REQUIRED_COLUMNS = ['id', 'A', 'B', 'C', 'D_Concept']


# -----------------------------------------------------------------------------
# Preparing and validating new items
# -----------------------------------------------------------------------------

def derive_dims(grid: str) -> Tuple[int, int]:
    """Square grids only, which is what the item format assumes."""
    n = len(str(grid))
    dim = int(round(n ** 0.5))
    return dim, dim


def validate_item(item: Dict) -> List[str]:
    """
    Return a list of problems with an item. Empty list means it is usable.

    These are the checks that decide whether an item is well formed and
    whether it is actually ambiguous -- which is the point of the design.
    """
    problems = []

    grids = {f: str(item.get(f, '')) for f in GRID_FIELDS}

    for name, g in grids.items():
        if not g or g == 'nan':
            problems.append(f'{name} is missing')
        elif not g.isdigit():
            problems.append(f'{name} contains non-digit characters')

    if problems:
        return problems

    lengths = {name: len(g) for name, g in grids.items()}
    if len(set(lengths.values())) > 1:
        problems.append(f'grids have different lengths: {lengths}')
        return problems

    n = next(iter(lengths.values()))
    dim = int(round(n ** 0.5))
    if dim * dim != n:
        problems.append(f'grid length {n} is not a perfect square')
        return problems

    # The literal answers. Both are derived, so they can be checked for free.
    d_pixel = get_d_matrix(item, 'pixel')
    d_row = get_d_matrix(item, 'row')

    if d_pixel == grids['D_Concept']:
        problems.append('D_Concept equals the pixel-level D_Matrix (not ambiguous)')
    if d_row == grids['D_Concept']:
        problems.append('D_Concept equals the row-level D_Matrix (not ambiguous)')

    if grids['D_Concept'] in (grids['A'], grids['B'], grids['C']):
        problems.append('D_Concept duplicates one of the input grids')

    # If the authoring tool exported its own D_Matrix, it should agree with
    # what get_d_matrix produces -- otherwise one of the two is wrong.
    # A blank or all-zero export is a placeholder, not a claim: D_Matrix is
    # generated procedurally from A, B and C, so there is nothing to check.
    exported = str(item.get('D_Matrix', '') or '').strip()
    is_placeholder = (
        not exported
        or exported.lower() == 'nan'
        or set(exported) == {'0'}
    )
    if not is_placeholder and exported not in (d_pixel, d_row):
        problems.append('exported D_Matrix matches neither the pixel nor row rule')

    return problems


def prepare_new_items(df: pd.DataFrame) -> List[Dict]:
    """
    Turn the imported dataframe into item dicts in the existing format.

    `xdim` / `ydim` are derived rather than read, and the exported
    `D_Matrix` is dropped: it is a pure function of A, B and C, so storing
    it invites drift. It is still used by `validate_item` as a consistency
    check before being discarded.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required column(s): {missing}')

    items = []
    for _, row in df.iterrows():
        item = {
            'id': str(row['id']),
            'A': str(row['A']),
            'B': str(row['B']),
            'C': str(row['C']),
            'D_Concept': str(row['D_Concept']),
            'concept': str(row['concept']) if 'concept' in df.columns else None,
        }
        xdim, ydim = derive_dims(item['A'])
        item['xdim'], item['ydim'] = xdim, ydim

        # Kept off the item, used only for validation.
        item['_exported_d_matrix'] = str(row.get('D_Matrix', '') or '')
        item['_problems'] = validate_item(item)
        items.append(item)

    return items


# -----------------------------------------------------------------------------
# Duplicate detection
# -----------------------------------------------------------------------------

def find_duplicates(item: Dict, existing: List[Dict]) -> List[Dict]:
    """
    Items in `existing` that share any identifying grid with `item`.

    Ids from an external tool are not comparable to the existing ones, so
    matching is on content. Each hit reports which fields matched: a match
    on all four is the same item, while a match on only A or only C is more
    likely a variant built from the same inputs, which you may still want.
    """
    hits = []
    for other in existing:
        matched = [
            f for f in GRID_FIELDS
            if str(item.get(f)) == str(other.get(f))
        ]
        if matched:
            hits.append({
                'id': other.get('id'),
                'concept': other.get('concept'),
                'matched_on': matched,
                'exact': len(matched) == len(GRID_FIELDS),
            })
    return hits


# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

def concept_table(items: List[Dict]) -> pd.DataFrame:
    """Items per concept by grid size, with row and column totals."""
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame([
        {'concept': i.get('concept'), 'size': f"{i.get('xdim')}x{i.get('ydim')}"}
        for i in items
    ])
    table = pd.crosstab(df['concept'], df['size'], margins=True, margins_name='Total')
    return table.sort_values('Total', ascending=False)


# -----------------------------------------------------------------------------
# Reviewer
# -----------------------------------------------------------------------------

class ItemReviewer:
    """
    Step through new items, inspect them, and accept the good ones.

    Parameters
    ----------
    new_items : pandas.DataFrame | list[dict]
        The imported items.
    existing_items : list[dict]
        The current item set. Never modified -- `self.items` starts as a
        deep copy of it.
    skip_duplicates : bool, default True
        Drop new items that exactly match an existing one before review.
    """

    def __init__(
        self,
        new_items: Union[pd.DataFrame, List[Dict]],
        existing_items: List[Dict],
        skip_duplicates: bool = True,
    ):
        if isinstance(new_items, pd.DataFrame):
            new_items = prepare_new_items(new_items)

        # The original list is never touched.
        self.existing = existing_items
        self.items = copy.deepcopy(existing_items)

        # Concepts already present. Accepting anything outside this set is a
        # new category and has to be confirmed, so a typo cannot quietly
        # create one.
        self.known_concepts = {
            i.get('concept') for i in existing_items if i.get('concept')
        }

        self.queue = []
        self.skipped_duplicates = []
        for item in new_items:
            dupes = find_duplicates(item, existing_items)
            item['_duplicates'] = dupes
            if skip_duplicates and any(d['exact'] for d in dupes):
                self.skipped_duplicates.append(item)
            else:
                self.queue.append(item)

        self.index = 0
        self.accepted_ids = []
        self.rejected_ids = []

    # -- state -------------------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self.queue)

    @property
    def current(self) -> Dict:
        if not self.queue:
            raise IndexError('Nothing left to review.')
        return self.queue[self.index]

    def report(self):
        """What came in, before reviewing anything."""
        n_new = len(self.queue)
        n_dup = len(self.skipped_duplicates)
        n_bad = sum(1 for i in self.queue if i['_problems'])
        n_partial = sum(1 for i in self.queue if i['_duplicates'])

        print(f'Existing items:        {len(self.existing)}')
        print(f'To review:             {n_new}')
        print(f'  with problems:       {n_bad}')
        print(f'  partial grid match:  {n_partial}')
        print(f'Exact duplicates skipped: {n_dup}')
        if n_dup:
            print('  ' + ', '.join(str(i['id']) for i in self.skipped_duplicates[:8]))
        concepts = pd.Series([i.get('concept') for i in self.queue])
        print('\nConcepts in the new items:')
        print(concepts.value_counts(dropna=False).to_string())

        unknown = {c for c in concepts.dropna().unique()
                   if c not in self.known_concepts and c not in ('None', 'nan')}
        if unknown:
            print(f"\nNot in the data yet: {', '.join(sorted(unknown))}")
            print('Accepting these needs new_concept=True.')

    # -- navigation --------------------------------------------------------

    def next(self, steps: int = 1):
        if not self.queue:
            print('Nothing left to review.')
            return
        self.index = (self.index + steps) % self.n_items
        return self.show()

    def previous(self, steps: int = 1):
        return self.next(-steps)

    def go_to(self, index: int):
        self.index = index % self.n_items
        return self.show()

    # -- decisions ---------------------------------------------------------

    def accept(
        self,
        concept: str = None,
        force: bool = False,
        new_concept: bool = False,
        advance: bool = True,
    ):
        """
        Add the current item to `self.items`.

        Parameters
        ----------
        concept : str, optional
            Overrides the concept from the CSV.
        force : bool, default False
            Accept even though validation found problems.
        new_concept : bool, default False
            Confirm a concept that is not yet in the item set. Without this,
            an unrecognised concept is refused, so a typo cannot silently
            create a category.
        """
        item = self.current

        if item['_problems'] and not force:
            print(f"Not accepted -- {item['id']} has problems:")
            for p in item['_problems']:
                print(f'  - {p}')
            print('Pass force=True to accept anyway.')
            return

        if concept is not None:
            item['concept'] = concept
        if not item.get('concept') or item['concept'] in ('None', 'nan'):
            print('No concept set. Pass one: reviewer.accept("concept_name")')
            return

        chosen = item['concept']
        if chosen not in self.known_concepts and not new_concept:
            close = [c for c in sorted(self.known_concepts)
                     if chosen.lower() in c.lower() or c.lower() in chosen.lower()]
            print(f"Concept '{chosen}' is not in the data. "
                  f'Would you like to add it as a new category?')
            print(f'  yes  ->  reviewer.accept("{chosen}", new_concept=True)')
            if close:
                print(f'  did you mean: {", ".join(close)}?')
            print(f'  existing concepts: {", ".join(sorted(self.known_concepts))}')
            return

        clean = {k: v for k, v in item.items() if not k.startswith('_')}
        self.items.append(clean)
        self.accepted_ids.append(item['id'])
        self.queue.pop(self.index)

        if chosen not in self.known_concepts:
            self.known_concepts.add(chosen)
            print(f"Added new concept '{chosen}'.")

        print(f"Accepted {clean['id']} as '{clean['concept']}'  "
              f"({len(self.accepted_ids)} accepted, {self.n_items} left)")

        if self.queue:
            self.index = self.index % self.n_items
            if advance:
                return self.show()
        else:
            print('Review complete.')

    def reject(self, advance: bool = True):
        item = self.current
        self.rejected_ids.append(item['id'])
        self.queue.pop(self.index)
        print(f"Rejected {item['id']}  "
              f"({len(self.rejected_ids)} rejected, {self.n_items} left)")

        if self.queue:
            self.index = self.index % self.n_items
            if advance:
                return self.show()
        else:
            print('Review complete.')

    def set_concept(self, concept: str):
        """Change the concept without accepting yet."""
        self.current['concept'] = concept
        print(f"Concept set to '{concept}' for {self.current['id']}")
        if concept not in self.known_concepts:
            print(f"  note: '{concept}' is not in the data -- accepting it "
                  f'will need new_concept=True')

    # -- display -----------------------------------------------------------

    def show(self, return_fig: bool = False):
        item = self.current
        xdim = item['xdim']

        # A malformed item cannot be drawn or have its literal answers
        # derived. Report the problems and stop rather than raising.
        if any(p.startswith(('grids have different', 'grid length',
                             'A is', 'B is', 'C is', 'D_Concept is'))
               or 'non-digit' in p for p in item['_problems']):
            print(f"Cannot draw {item['id']} -- malformed:")
            for p in item['_problems']:
                print(f'  - {p}')
            print('Use reviewer.reject() to drop it, or .next() to skip.')
            return

        d_pixel = get_d_matrix(item, 'pixel')
        d_row = get_d_matrix(item, 'row')

        fig = plt.figure(figsize=(9, 8.5))
        gs = fig.add_gridspec(
            3, 3, width_ratios=[1, 0.35, 1], height_ratios=[1, 1, 1.15], hspace=0.55
        )

        self._grid(fig.add_subplot(gs[0, 0]), item['A'], 'A', xdim)
        self._arrow(fig.add_subplot(gs[0, 1]))
        self._grid(fig.add_subplot(gs[0, 2]), item['B'], 'B', xdim)

        self._grid(fig.add_subplot(gs[1, 0]), item['C'], 'C', xdim)
        self._arrow(fig.add_subplot(gs[1, 1]))
        self._grid(fig.add_subplot(gs[1, 2]), item['D_Concept'], 'D Concept', xdim)

        bottom = gs[2, :].subgridspec(1, 3, wspace=0.45)
        self._grid(fig.add_subplot(bottom[0, 0]), d_pixel, 'D Matrix (pixel)', xdim)
        self._grid(fig.add_subplot(bottom[0, 1]), d_row, 'D Matrix (row)', xdim)

        ax = fig.add_subplot(bottom[0, 2])
        ax.axis('off')
        notes = []
        if d_pixel == d_row:
            notes.append('pixel == row\n(no pixel/row contrast)')
        if item['_duplicates']:
            for d in item['_duplicates'][:3]:
                notes.append(f"matches {d['id']}\non {', '.join(d['matched_on'])}")
        ax.text(0.5, 0.55, '\n\n'.join(notes) if notes else 'No overlap\nwith existing items',
                ha='center', va='center', fontsize=9,
                color='firebrick' if notes else 'dimgray')

        title = (f"{self.index + 1}/{self.n_items}  |  {item['id']}  |  "
                 f"{item.get('concept')}  |  {xdim}x{item['ydim']}")
        colour = 'firebrick' if item['_problems'] else 'black'
        fig.suptitle(title, fontsize=13, fontweight='bold', color=colour)

        if item['_problems']:
            print(f"Problems with {item['id']}:")
            for p in item['_problems']:
                print(f'  - {p}')

        if return_fig:
            plt.close(fig)
            return fig
        plt.show()

    @staticmethod
    def _grid(ax, grid, title, xdim):
        plot_matrix(convert_to_array(str(grid), xdim), title=title, ax=ax, size=11)

    @staticmethod
    def _arrow(ax):
        ax.axis('off')
        ax.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', linewidth=2))

    # -- output ------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Items per concept by grid size, for the updated list."""
        return concept_table(self.items)

    def save(self, path: str):
        """Write the updated list. Does not touch the original file."""
        with open(path, 'w') as f:
            json.dump(self.items, f, indent=1)
        print(f'Wrote {len(self.items)} items to {path} '
              f'({len(self.accepted_ids)} newly accepted)')