import random
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
from typing import *

# Load data
items = json.load(open('data/human_data_collection/AmbigousARC.json'))

# Constants
TASKS = ["generation", "discrimination", "recognition"]
ROW_PIXEL_CONCEPTS = ["keep_above_or_below"]


def participant_item_splits(
    items_: List[Dict],
    n_unique: Dict[str, int],                         
    n_shared: int,
    n_concepts: int,
    row_pixel_n: int
) -> Dict[str, List[Dict]]:

    n_items_per_concept = {}
    for task in TASKS:
        n_items_per_concept[task] = n_unique[task] // n_concepts

    # shuffle items
    items = copy.deepcopy(items_)
    random.shuffle(items)

    # split items into main, row/pixel and mirror
    items_main = [item for item in items if int(item['item_id'][-1]) == 1]
    items_row_pixel = [item for item in items_main if item['concept'] in ROW_PIXEL_CONCEPTS]
    items_mirror = [item for item in items if int(item['item_id'][-1]) != 1]

    # Remove row/pixel items from main
    items_main = [item for item in items_main if item['concept'] not in ROW_PIXEL_CONCEPTS]

    # Allocate row/pixel items
    row_pixel_split = {
        "generation": items_row_pixel[:row_pixel_n],
        "discrimination": items_row_pixel[row_pixel_n : row_pixel_n + row_pixel_n*2], 
        "recognition": items_row_pixel[row_pixel_n + row_pixel_n*2 : row_pixel_n + row_pixel_n*3]
    }

    # Split discrimination items into pixel and row
    for idx, item in enumerate(row_pixel_split["discrimination"]):
        if idx < row_pixel_n:
            item['matrix_level'] = 'pixel'
        else:
            item['matrix_level'] = 'row'
            item['D_Matrix'] = item['D_Matrix_Row']

    # assign random concepts to each task
    concepts = list(set([item['concept'] for item in items_main]))
    concepts_per_task = defaultdict(list)
    for task in TASKS:
        # exclude row/pixel concepts as they are already assigned
        selected_concepts = random.sample([c for c in concepts if c not in ROW_PIXEL_CONCEPTS], n_concepts)    
        concepts_per_task[task] = selected_concepts 

    # Step 2: Generate unique and shared items for each task
    shared_items = items_main[:n_shared]
    unique_items = items_main[n_shared:]

    # Assign unique, shared and mirror items to each task
    task_items = defaultdict(list)
    for task_idx, task in enumerate(TASKS):
        # Add row/pixel items
        task_items[task].extend(row_pixel_split[task])

        # Add shared items (mirror for each task)
        for shared_item in shared_items:
            mirror_item = [item for item in items_mirror if item['item_id'] == f'{shared_item['main_id']}_{task_idx+2}']
            task_items[task].extend(mirror_item)
        
        # Add unique items
        for concept in concepts_per_task[task]:
            concept_items = [item for item in unique_items if item['concept'] == concept]
            task_items[task].extend(concept_items[:n_items_per_concept[task]])

            # Remove assigned items
            unique_items = [item for item in unique_items if item not in concept_items[:n_items_per_concept[task]]]

    # Final shuffle
    for task in TASKS:
        random.shuffle(task_items[task])
    
    return task_items


# Human vs AI comparisons
# - Task Performance (generation, discrimination, recognition)
# - Concept Performance (11 concepts)
# - Row vs Pixel Literal solutions
# - Item Performance 
#   * Are items predictive across tasks?
#   * Specific items where participants choose conceptual/literal answers (discrimination)


### Simulation parameters
n_participants = 100


### Item parameters
# Unique per task
n_unique = dict(
    generation = 15,
    discrimination = 20,
    recognition = 20
)

# Shared between tasks
n_shared = 5 # Shared items between tasks
n_concepts = 10
row_pixel_n = 2


### Estimate time to complete tasks
gen_time_to_complete = 45/30 # 45 minutes to complete 30 tasks [open-ended items] = 1,5 minutes per item
mc_time_to_complete = 0.33 # 45 seconds per item [multiple choice items]
cost_per_minute = 10 / 60 # 10 euros per hour

n_items_gen = (n_unique['generation']//n_concepts)*n_concepts+n_shared+row_pixel_n
n_items_disc = n_unique['discrimination']+n_shared+(row_pixel_n*2)
n_items_rec = n_unique['recognition']+n_shared+row_pixel_n
gen_time = n_items_gen * gen_time_to_complete
dis_time = n_items_disc * mc_time_to_complete
rec_time = n_items_rec * mc_time_to_complete
total_time = gen_time + dis_time + rec_time

print(f"Time to complete generation task: {int(gen_time)} minutes ({n_items_gen} items)")
print(f"Time to complete discrimination task: {int(dis_time)} minutes ({n_items_disc} items)")  
print(f"Time to complete recognition task: {int(rec_time)} minutes ({n_items_rec} items)")
print(F'Total time to complete all tasks per participant: {int(total_time)} minutes')
print(f"Cost per participant: {int(total_time*cost_per_minute)} euros")


### SIMULATE DATA
concepts_sub = {task: [] for task in TASKS}
items_sub = {task: [] for task in TASKS}

all_items_n = []
for i in range(n_participants):
    task_items = participant_item_splits(items, n_unique, n_shared, n_concepts, row_pixel_n)
    
    all_items = []
    for task in task_items:
        df = pd.DataFrame(task_items[task])
        all_items.append(len(df))
        concepts_sub[task].extend(list(df['concept'].unique()))
        items_sub[task].extend(list(df['main_id'].unique()))
    all_items_n.append(sum(all_items))
    
print(f"Mean items per participant: {int(sum(all_items_n)/n_participants)}")

### Count concepts and items
concept_count = {task: pd.Series(concepts_sub[task]).value_counts() for task in concepts_sub}
print("\nConcepts:")
plt.subplots(3, 1, figsize=(5, 15))
for task in concept_count:
    print(f"\nTask: {task}")
    print(f"Mean concepts per participant: {int(concept_count[task].mean())}")
    plt.subplot(3, 1, TASKS.index(task)+1)
    concept_count[task].plot(kind='bar', title=task)
    plt.xticks([])
    plt.xlabel('Concepts')
    plt.ylabel('N participants')

item_count = {task: pd.Series(items_sub[task]).value_counts() for task in items_sub}
print("\nItems:")
plt.subplots(3, 1, figsize=(5, 15))
for task in item_count:
    print(f"\nTask: {task}")
    print(f"Mean items per participant: {int(item_count[task].mean())}")
    plt.subplot(3, 1, TASKS.index(task)+1)
    item_count[task].hist(bins=20)
    plt.title(task)
    plt.ylabel('N Items')
    plt.xlabel('N participants')


# Exclusion criteria
# - Min time per item (1s)
# - Max time per item
# - 80% of items complete 
# - above chance level performance multiple choice items
# - minimum % correct generation (old study) - look at responses below 2 std from mean

# Add an attention check (?)