#%% Imports
import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset
from utils.plot_utils import plot_item, get_percentage_ticks, plot_correlation_matrix
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.dpi'] = 300

run_dir = 'run_2024-11-12' 
generation = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

# %%
for idx, task in enumerate(['generation', 'discrimination']):
    response_cols = ['concept_response', 'matrix_response', 'duplicate_response']

    means = eval(task).df.groupby('concept')[response_cols].mean()
    sems = eval(task).df.groupby(['concept', 'model'])[response_cols].sem().groupby('concept').mean()

    # Sort the DataFrame
    if idx == 0:
        means = means.sort_values(by='concept_response', ascending=True)
        index = means.index
    else:
        means = means.loc[index]

    sems = sems.loc[means.index]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 20))
    means.plot(
        kind='barh', 
        ax=ax, 
        color=['#d44e86', '#665190', '#adf4dc', '#f4ad42'], 
        edgecolor='black', 
        xerr=sems, 
        #capsize=5
    )
    plt.ylabel('')
    plt.xticks(*get_percentage_ticks(means), size=18)
    plt.yticks(size=18)
    plt.xlabel('% of Responses', size=18)
    plt.legend(fontsize=18, edgecolor='none', loc='lower right')
    plt.show()

# %%
tasks_df = pd.DataFrame({
    'Discrimination': discrimination.df.groupby('concept')['concept_response'].mean(),
    'Recognition': recognition.df.groupby('concept')['concept_response'].mean(),
    'Generation': generation.df.groupby('concept')['concept_response'].mean(),
})
task_df_sem = pd.DataFrame({
    'Discrimination': discrimination.df.groupby(['concept', 'model'])['concept_response'].sem().groupby('concept').mean(),
    'Recognition': recognition.df.groupby(['concept', 'model'])['concept_response'].sem().groupby('concept').mean(),
    'Generation': generation.df.groupby(['concept', 'model'])['concept_response'].sem().groupby('concept').mean(),
})
tasks_df = tasks_df.loc[means.index]
task_df_sem = task_df_sem.loc[means.index]
# Plotting
fig, ax = plt.subplots(figsize=(10, 20))
tasks_df.plot(
    kind='barh', 
    ax=ax, 
    color=['#d44e86', '#665190', '#adf4dc'], 
    edgecolor='black', 
    xerr=task_df_sem,
)
plt.ylabel('')
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel('Concept Responses', size=18)
plt.title(f'{task.capitalize()} Task', size=20)
plt.legend(fontsize=18, edgecolor='none', loc='lower right')
plt.show()


plot_correlation_matrix(tasks_df[['Generation', 'Discrimination', 'Recognition']].corr(), vmin=-1, vmax=1)

plt.plot(tasks_df['Generation'], tasks_df['Discrimination'], 'o')


#%%
task = 'recognition'
response_cols = ['concept_response']

means = eval(task).df.groupby('concept')[response_cols].mean()
sems = eval(task).df.groupby(['concept', 'model'])[response_cols].sem().groupby('concept').mean()

# Sort the DataFrame
if idx == 0:
    means = means.sort_values(by='concept_response', ascending=True)
    index = means.index
else:
    means = means.loc[index]

sems = sems.loc[means.index]

# Plotting
fig, ax = plt.subplots(figsize=(10, 20))
means.plot(
    kind='barh', 
    ax=ax, 
    color=['#d44e86', '#665190', '#adf4dc', '#f4ad42'], 
    edgecolor='black', 
    xerr=sems, 
    #capsize=5
)
plt.ylabel('')
plt.xticks(*get_percentage_ticks(means), size=18)
plt.yticks(size=18)
plt.xlabel('% of Responses', size=18)
plt.legend(fontsize=18, edgecolor='none', loc='lower right')
plt.show()
#%%
# confusion matrix for concepts
import numpy as np
from itertools import product

# Extract unique concepts and map them to indices
concepts = recognition.df['concept'].unique()
concept_to_index = {concept: idx for idx, concept in enumerate(concepts)}

# Initialize the confusion matrix
confusion_matrix = {
    model: np.zeros((len(concepts), len(concepts)), dtype=int)
    for model in recognition.df['model'].unique()
}

# Populate the confusion matrix
for model in recognition.df['model'].unique():
    for i, j in product(concepts, repeat=2):
        i_idx = concept_to_index[i]
        j_idx = concept_to_index[j]
        confusion_matrix[model][i_idx, j_idx] = recognition.df[
            (recognition.df['model'] == model) &
            (recognition.df['concept'] == i) &
            (recognition.df['choice'] == j)
        ].shape[0]

def plot_confusion_matrix(mat, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap='Blues')
    plt.colorbar(label='Counts')

    # Add labels to axes
    plt.xticks(ticks=range(len(concepts)), labels=concepts, rotation=45, ha='right')
    plt.yticks(ticks=range(len(concepts)), labels=concepts)
    plt.xlabel('Predicted Concepts')
    plt.ylabel('True Concepts')
    plt.title(f'Confusion Matrix for Model {title}')

    # Add counts to each cell
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            plt.text(j, i, mat[i, j], ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

avg_confusion_mat = np.mean([confusion_matrix[model] for model in confusion_matrix], axis=0).astype(int)

plot_confusion_matrix(avg_confusion_mat, 'Recognition')
# %%