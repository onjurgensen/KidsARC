#%% Imports
import os
import json
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from utils.globals import *
from utils.eval import Eval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
from utils.human_preprocessing import df as human_df

from sklearn.metrics import cohen_kappa_score

old_arc = False
run_dir = 'old/2024-11-12/run_2024-11-12' if old_arc else 'run_2024-11-12'
recognition = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'), old_arc=old_arc)
recognition_r1 = Eval(os.path.join(RESULTS_DIR, 'reasoning', 'recognition.json'), old_arc=old_arc)

# %%
task = 'Recognition'
unique_items = recognition.df['item_main_id'].unique()

# Majority vote
model_choices_df = []
for model in recognition.models_names:
    for item in unique_items:   
        df_item = recognition.df[(recognition.df['model'] == model) & (recognition.df['item_main_id'] == item)]
        model_choices_df.append(
            {
                'model': model,
                'item_main_id': item,
                'choice': df_item['choice'].mode().values[0]
            }
        )
model_choices_df = pd.DataFrame(model_choices_df)
model_choices_df = pd.concat([model_choices_df, recognition_r1.df[['model', 'item_main_id', 'choice']]])  # Add missing items
model_names = model_choices_df['model'].unique()


cohen = []
# Participant vs Model
for sub in human_df['participant_fk'].unique():
    df_sub = human_df[(human_df['participant_fk'] == sub) & (human_df['task'] == task)]
    sub_choices = df_sub[['main_id', 'response']].sort_values(by='main_id')
    for model in model_names:
        model_choices = model_choices_df[
            (model_choices_df['model'] == model) &
            (model_choices_df['item_main_id'].isin(df_sub['main_id']))
        ].sort_values(by='item_main_id')

        # Cohen's Kappa
        if len(sub_choices) > 2:
            ck = cohen_kappa_score(sub_choices['response'], model_choices['choice'])
        else:
            ck = np.nan
        cohen.append({'system_1': sub, 'system_2': model, 'kappa': ck})
        cohen.append({'system_1': model, 'system_2': sub, 'kappa': ck})

    # Participant vs Participant
    for sub_2 in human_df['participant_fk'].unique():
        if sub == sub_2:
            cohen.append({'system_1': sub, 'system_2': sub_2, 'kappa': 1})
            continue
        df_sub_2 = human_df[(human_df['participant_fk'] == sub_2) & (human_df['task'] == task) & (human_df['main_id'].isin(df_sub['main_id']))]
        sub_2_choices = df_sub_2[['main_id', 'response']].sort_values(by='main_id')
        sub_1_choices = sub_choices[sub_choices['main_id'].isin(df_sub_2['main_id'])].sort_values(by='main_id')
        if len(sub_1_choices) > 2:
            ck = cohen_kappa_score(sub_1_choices['response'], sub_2_choices['response'])
        else:
            ck = np.nan
        cohen.append({'system_1': sub, 'system_2': sub_2, 'kappa': ck})

# Model vs Model
for model_1 in model_names:
    for model_2 in model_names:
        if model_1 == model_2:
            cohen.append({'system_1': model_1, 'system_2': model_2, 'kappa': 1} )
            continue
        df_model_1 = model_choices_df[model_choices_df['model'] == model_1].sort_values(by='item_main_id')
        df_model_2 = model_choices_df[model_choices_df['model'] == model_2].sort_values(by='item_main_id')
        ck = cohen_kappa_score(df_model_1['choice'], df_model_2['choice'])
        cohen.append({'system_1': model_1, 'system_2': model_2, 'kappa': ck})

cohen = pd.DataFrame(cohen)

# %%
# Visualize matrix
# Create pivot table
import numpy as np
import matplotlib.pyplot as plt

# Create pivot table
cohen_matrix = cohen.pivot(index='system_1', columns='system_2', values='kappa')

# Sort participants by accuracy
human_df_index = human_df.groupby('participant_fk')['concept_answer'].mean().sort_values(ascending=False).index#[:40]
#human_df_index = np.random.shuffle(human_df_index)
cohen_matrix = cohen_matrix.reindex(list(human_df_index) + list(model_names), axis=0)
cohen_matrix = cohen_matrix.reindex(list(human_df_index) + list(model_names), axis=1)

# Mask NaN values
masked_matrix = np.ma.masked_invalid(cohen_matrix.values)

# Visualize with heatmap
plt.figure(figsize=(12, 12))
# Use masked array for displaying NaN as white
plt.imshow(masked_matrix, cmap='viridis', interpolation='nearest')

# Add gridlines and title
plt.title("Cohen's Kappa Agreement Matrix")
plt.xticks(range(len(cohen_matrix.columns)), [get_model_name(i) for i in cohen_matrix.columns], rotation=90, ha='right')
plt.yticks(range(len(cohen_matrix.index)), [get_model_name(i) for i in cohen_matrix.index])

# Add a colorbar
cbar = plt.colorbar(shrink=0.75, aspect=20)  # Shrink and adjust the aspect ratio
cbar.set_label("Cohen's Kappa", fontsize=14)

# Show plot
plt.show()
# %%
