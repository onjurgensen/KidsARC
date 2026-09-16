#%% Imports
import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset
from utils.plot_utils import plot_item
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind
from rich import print as rprint

#%%
files = sorted(os.listdir(os.path.join(RESULTS_DIR, 'example_test')))
df_example = pd.DataFrame({})
for f in files:
    task = f.split('_')[0]
    concept = f.split('_')[1]
    e = Eval(os.path.join(RESULTS_DIR, 'example_test', f))
    df = e.df.groupby('model')['concept_response'].mean().sort_index().reset_index()
    df['task'] = task
    df['concept'] = concept
    df['model'] = df['model'].apply(lambda x: get_model_name(x))
    df_example = pd.concat([df_example, df])
    
# %%
df = df_example.groupby(['task', 'concept'])['concept_response'].mean().reset_index()
df_sem = df_example.groupby(['task', 'concept', 'model'])['concept_response'].mean().groupby(['task', 'concept']).sem().reset_index()
plt.figure(figsize=(10, 6))
colors = ['#d44e86', '#665190', '#adf4dc']
for i, (task, group) in enumerate(df.groupby('task')):
    plt.subplot(1, 3, i+1)
    for j, (concept, row) in enumerate(group.iterrows()):
        sem = df_sem[(df_sem['task'] == task) & (df_sem['concept'] == row['concept'])]
        plt.bar(j, row['concept_response'], color=colors[j], edgecolor='black', yerr=sem['concept_response'])
    plt.title(task)
    plt.xticks(range(j+1), ['Different Concept', 'No Example', 'Same Concept'], rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%%
df = df_example[df_example['task'] == 'discrimination']
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    plt.figure(figsize=(10, 6))
    for i, (concept, group) in enumerate(model_data.groupby('concept')):
        plt.bar(i, group['concept_response'], label=concept)
    plt.title(model)
    plt.xticks(range(i+1), model_data['concept'].unique())
    plt.legend(title='Concept')
    plt.show()

dataset = AmbigousARCDataset(task='generation', example_item='same')
dataset.plot(dataset.example_items[4])
dataset.plot(4)