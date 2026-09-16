#%% Imports
import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.plot_utils import plot_item
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

discrimination_color = Eval(os.path.join(RESULTS_DIR, 'run_2024-11-12', 'discrimination.json'))
discrimination_example = Eval(os.path.join(RESULTS_DIR, 'run_example_2024-11-12', 'discrimination.json'))
discrimination_answeropt = Eval(os.path.join(RESULTS_DIR, 'run_answeropt_2024-11-12', 'discrimination.json'))


# %%
item_df_color = discrimination_color.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df_example = discrimination_example.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df_answeropt = discrimination_answeropt.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()

plt.figure(figsize=(10, 5))
item_df_color.sort_values('concept_response', ascending=False).plot(kind='bar', stacked=True)
plt.xticks([])
plt.show()

plt.figure(figsize=(10, 5))
item_df_example.sort_values('concept_response', ascending=False).plot(kind='bar', stacked=True)
plt.xticks([])
plt.show()

#item_df_color.hist(bins=20, figsize=(10, 5))
#item_df_example.hist(bins=20, figsize=(10, 5))

def plot_most_common(item_df, choice, top=3):
    item_df_sorted = item_df.sort_values(by=f'{choice}_response', ascending=False)[:top]
    for i in range(top):    
        print(f'Item: {item_df_sorted.index[i]}')
        concept = discrimination_color.dataset.get_item_by_id(item_df_sorted.index[i]+'_1')['concept']
        discrimination_color.dataset.plot(item_df_sorted.index[i], f'Concept: {concept}\n{item_df_sorted.iloc[i][f'{choice}_response'] * 100:.2f}%')

# Item df across both color and example mirrors
item_df_all = pd.concat([item_df_color, item_df_example, item_df_answeropt]).groupby(level=0).mean()
plot_most_common(item_df_all, 'matrix', top=10)

#histograms 
item_df_all['matrix_response'].hist()


discrimination_example.df[discrimination_example.df['item_main_id'] == 'PIle2W']['matrix_response'].mean()  
item_df_example[item_df_example.index == 'PIle2W']

item_df_all.sort_values('matrix_response', ascending=False).head(3)


discrimination_color.dataset.plot('zjvdIo')

d = pd.concat([item_df_color, item_df_example, item_df_answeropt]).sort_index()
# std over rows

grouped = d.groupby('item_main_id')

# Calculate the range (max - min) for each column within each group
def calculate_range(group):
    ranges = group.max() - group.min()  # Calculate range for each column
    return ranges

d[d.index == 'zjvdIo']

# Apply the range calculation to each group
ranges = grouped.apply(calculate_range)

# Identify item_main_ids with high variance (e.g., range > 0.5 in any column)
threshold = 0.5
high_variance_items = ranges[ranges.max(axis=1) > threshold]

# Display the results
print("Item_main_ids with high variance across rows:")
print(high_variance_items)
# %%
