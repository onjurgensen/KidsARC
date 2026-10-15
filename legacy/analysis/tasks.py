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


def plot_correlation_matrix(corr_mat, vmin=-1, vmax=1, annot=True, square=True, lower=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mask = np.tril(corr_mat) == 0 if lower else None

    heatmap = sns.heatmap(
        corr_mat,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        square=square,
        mask=mask,
        linewidths=0,  # Disable bounding boxes for all cells
        annot_kws={"size": 12},  # Increase annotation font size
        # italic
        cbar_kws={'label': 'r'},  # Add colorbar label
        ax=ax,  # Use provided Axes
    )

    # Increase the font size of the colorbar ticks
    heatmap.figure.axes[-1].tick_params(labelsize=14)
    heatmap.figure.axes[-1].yaxis.label.set_size(16)
    # vertical 
    heatmap.figure.axes[-1].yaxis.label.set_rotation(0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)

    # Add bounding boxes only for non-masked cells
    if lower:
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                if not mask[i, j]:  # Only if the cell is not masked
                    rect = plt.Rectangle(
                        (j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.5
                    )
                    ax.add_patch(rect)
    if ax is None:
        plt.show()

old_arc = False
run_dir = 'old/2024-11-12/run_2024-11-12' if old_arc else 'run_2024-11-12'
generation = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'), old_arc=old_arc)
discrimination = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'), old_arc=old_arc)
recognition = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'), old_arc=old_arc)

generation_example = Eval(os.path.join(RESULTS_DIR, 'run_example_2024-11-12', 'generation.json'), old_arc=old_arc)
discrimination_example = Eval(os.path.join(RESULTS_DIR, 'run_example_2024-11-12', 'discrimination.json'), old_arc=old_arc)
recognition_example = Eval(os.path.join(RESULTS_DIR, 'run_example_2024-11-12', 'recognition.json'), old_arc=old_arc)

generation_r1 = Eval(os.path.join(RESULTS_DIR, 'reasoning', 'generation.json'), old_arc=old_arc)
discrimination_r1 = Eval(os.path.join(RESULTS_DIR, 'reasoning', 'discrimination.json'), old_arc=old_arc)
recognition_r1 = Eval(os.path.join(RESULTS_DIR, 'reasoning', 'recognition.json'), old_arc=old_arc)

#excl = (discrimination.df.groupby('model')['same_as_example'].mean() > 0.3).apply(lambda x: x if x else None).dropna().index.to_list()

# %% Task Performance
df_means = pd.DataFrame({
    'Discrimination': [],
    'Recognition': [],
    'Generation': []
})

raw_accs = {}
for task in ['generation', 'discrimination', 'recognition']:
    df_example = eval(f'{task}_example.df')
    df_example['mirror'] = df_example['mirror'].astype(int) + 5
    full_df = pd.concat([eval(f'{task}.df'), df_example])
    full_df_concept = full_df.groupby(['model', 'mirror'])['concept_response'].mean().unstack()
    df_means[task.capitalize()] = full_df_concept.mean(axis=1)
    raw_accs[task.capitalize()] = full_df_concept.T.to_dict('list')


# add human performance
df_means.loc['Human'] = (human_df.groupby('task')['concept_answer'].sum() / human_df.groupby('task').size())
for task in ['Generation', 'Discrimination', 'Recognition']:
    raw_accs[task]['Human'] = human_df[human_df['task'] == task].groupby('participant_fk')['concept_answer'].mean().values

# add r1 performance
for task in ['generation', 'discrimination', 'recognition']:
    df_means.loc['R1', task.capitalize()] = eval(f'{task}_r1').df['concept_response'].mean()
    raw_accs[task.capitalize()]['R1'] = None

# Sort the DataFrame
df_means = df_means.sort_values(by=['Generation'], ascending=True)# Sort the DataFrame

# Model names adjustment
model_names = [get_model_name(name) for name in df_means.index]

# Plotting
# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Number of models and tasks
n_models = len(df_means.index)
n_tasks = len(df_means.columns)
width = 0.25  # Width of each bar

# Create x-coordinates for each group of bars
x = np.arange(n_models)

# Plot each task's data
tasks = ['Discrimination', 'Recognition', 'Generation']
colors = ['#d44e86', '#665190', '#adf4dc']  # Different color for each task

# Plot bars and raw data for each task
for i, task in enumerate(tasks):
    # Plot mean values as bars
    pos = x + (i - 1) * width
    plt.plot(
        pos, 
        df_means[task], 
        marker='D', 
        markersize=10, 
        color=colors[i], 
        alpha=1,
        label=task,
        zorder=2,
        linestyle='None',
    markeredgecolor='black',  # Black outline
    markeredgewidth=1.5       # Outline thickness
    )
    
    # Plot raw data points
    for j, model in enumerate(df_means.index):
        raw_data = raw_accs[task][model]
        if raw_data is None:
            continue
        if isinstance(raw_data, np.ndarray):  # Handle Human data
            plt.scatter(
                np.repeat(pos[j], len(raw_data)), 
                        raw_data, color=colors[i], alpha=0.1, s=20)
        else:
            plt.scatter(
                np.repeat(pos[j], len(raw_data)), 
                raw_data, color=colors[i], alpha=0.3, s=20)

# Customize the plot
plt.xlabel('Models')
plt.ylabel('Accuracy')
#plt.title('Model Performance by Task\n(Bars show means, points show raw data)')
plt.xticks(x, model_names, rotation=45, ha='right')
plt.legend()


# Set axis labels and ticks
plt.ylabel('Conceptual Responses', size=16)
plt.xticks(np.arange(0, len(df_means), 1), model_names, size=14, rotation=45, ha='right')
plt.yticks(np.linspace(0, 1, num=5), [f'{x:.0%}' for x in np.linspace(0, 1, num=5)], size=14)
plt.xlabel('')
plt.ylim(0, 1.05)

# # Calculate overall means
# mean_discrimination = df_means['Discrimination'].mean()
# mean_recognition = df_means['Recognition'].mean()
# mean_generation = df_means['Generation'].mean()

# # Overlay circles on the y-axis for the overall means
# plt.scatter(-0.5, mean_discrimination, color='#d44e86', s=300, zorder=0, edgecolor='black', marker='>')
# plt.scatter(-0.5, mean_recognition, color='#665190', s=350, zorder=0, edgecolor='black', marker='>')
# plt.scatter(-0.5, mean_generation, color='#adf4dc', s=350, zorder=0, edgecolor='black', marker='>')

# Customize the legend
plt.legend(fontsize=14, edgecolor='none')

# Remove unnecessary spines
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)

#plt.axhline(y=1/4, color='grey', linestyle='--', linewidth=1, zorder=0)

# Show the plot
#plt.tight_layout()
#plt.savefig('AmbigousARC_acl_2025/plots/task_performance.svg')
plt.show()

#%%
# Correlation between tasks
df_con = pd.DataFrame({
    'Recognition': recognition.df.groupby('model')['concept_response'].mean(),
    'Generation': generation.df.groupby('model')['concept_response'].mean(),
    'Discrimination': discrimination.df.groupby('model')['concept_response'].mean(),
})
df_con_human = (human_df.groupby(['participant_fk', 'task'])['concept_answer'].sum() / human_df.groupby(['participant_fk', 'task']).size()).unstack()
corr_mat = df_con.corr()
corr_mat_human = df_con_human.corr()
plot_correlation_matrix(corr_mat, vmin=0, vmax=1)
plot_correlation_matrix(corr_mat_human, vmin=0, vmax=1)

#%%
# cohens kappa
from sklearn.metrics import cohen_kappa_score

def mcc(array1, array2):
    array1, array2 = np.array(array1), np.array(array2)
    
    # Compute confusion matrix components
    TP = np.sum((array1 == 1) & (array2 == 1))  # True Positives
    TN = np.sum((array1 == 0) & (array2 == 0))  # True Negatives
    FP = np.sum((array1 == 0) & (array2 == 1))  # False Positives
    FN = np.sum((array1 == 1) & (array2 == 0))  # False Negatives
    
    # Compute MCC using the formula
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    # Avoid division by zero
    return numerator / denominator if denominator > 0 else np.nan # MCC is 0 if undefined

def jaccard_index(array1, array2):
    array1, array2 = np.array(array1), np.array(array2)
    intersection = np.sum((array1 == 1) & (array2 == 1))
    union = np.sum((array1 == 1) | (array2 == 1))
    return intersection / union if union != 0 else np.nan 


same_tasks = human_df['itemid'].apply(lambda x: str(x).split('_')[1] != '1')
same_tasks_df = human_df[same_tasks].reset_index(drop=True)
human_df_index = human_df.groupby('participant_fk')['concept_answer'].mean().sort_values(ascending=False).index#[:40]

corrs = []
for sub in human_df_index:
    sub_df = same_tasks_df[same_tasks_df['participant_fk'] == sub]
    performance = []
    for task in ['Generation', 'Recognition']:
        sub_task_df = sub_df[sub_df['task'] == task].sort_values(by='main_id')
        performance.append(sub_task_df['concept_answer'].values)
    ck = mcc(*performance)
    corrs.append(ck)

print(np.nanmean(corrs))
print(np.nanstd(corrs))

plt.hist(corrs, bins=20)


#%% Response Types
response_cols = ['concept_response', 'matrix_response', 'duplicate_response']
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

df_gen = generation.df.groupby('model')[response_cols].mean().sort_values('concept_response', ascending=True)
df_gen.plot(kind='bar', stacked=False, color=['#B53A3D', '#508FA3', '#F3E4D0'], ax=ax[0], edgecolor='black')
ax[0].set_title('Generation', fontsize=16)
ax[0].set_ylabel('% of responses', fontsize=14)
ax[0].set_yticks(np.arange(0, 0.5, 0.1), ['{:.0f}%'.format(i*100) for i in np.arange(0, 0.5, 0.1)], fontsize=12)
ax[0].set_xticks(range(df_gen.shape[0]), [get_model_name(i) for i in df_gen.index], rotation=45, ha='right', fontsize=12)
ax[0].set_xlabel('')

ax[0].legend(['Concept', 'Matrix', 'Duplicate'], frameon=False, fontsize=12)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)


df_dis = discrimination.df.groupby('model')[response_cols].mean().sort_values('concept_response', ascending=True)
df_dis.plot(kind='bar', stacked=False, color=['#B53A3D', '#508FA3', '#F3E4D0'], edgecolor='black', ax=ax[1])
ax[1].set_title('Discrimination', fontsize=16)
ax[1].set_ylabel('% of responses', fontsize=14)
ax[1].set_yticks(np.arange(0, 0.9, 0.2), ['{:.0f}%'.format(i*100) for i in np.arange(0, 0.9, 0.2)], fontsize=12)
ax[1].set_xticks(range(df_dis.shape[0]), [get_model_name(i) for i in df_dis.index], rotation=45, ha='right', fontsize=12)
ax[1].set_xlabel('')
ax[1].legend(['Concept', 'Matrix', 'Duplicate'], frameon=False, fontsize=12)
ax[1].legend([], frameon=False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

plt.show()

np.corrcoef(df_gen['concept_response'], df_gen['duplicate_response'])
np.corrcoef(df_dis['concept_response'], df_dis['duplicate_response'])


discrimination.df

#%%
import numpy as np

def mcc(array1, array2):
    array1, array2 = np.array(array1), np.array(array2)
    
    # Compute confusion matrix components
    TP = np.sum((array1 == 1) & (array2 == 1))  # True Positives
    TN = np.sum((array1 == 0) & (array2 == 0))  # True Negatives
    FP = np.sum((array1 == 0) & (array2 == 1))  # False Positives
    FN = np.sum((array1 == 1) & (array2 == 0))  # False Negatives
    
    # Compute MCC using the formula
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    # Avoid division by zero
    return numerator / denominator if denominator > 0 else np.nan # MCC is 0 if undefined

# Example usage
array1 = np.array([1, 0, 0, 0, 0])
array2 = np.array([1, 1, 1, 1, 1])

mcc_value = mcc(array1, array2)
print(f"MCC: {mcc_value:.4f}")  # Should print -1.0000 (perfect inverse correlation)

corr_mats = []
for i, model in enumerate(df_con.index.unique()):
    discrimination_model = discrimination.df[discrimination.df['model'] == model]
    generation_model = generation.df[generation.df['model'] == model]
    recognition_model = recognition.df[recognition.df['model'] == model]
    df_item_concept = pd.DataFrame({
        'Discrimination': discrimination_model['concept_response'],
        'Generation': generation_model['concept_response'],
        'Recognition': recognition_model['concept_response'],
    })
    corr_mats.append(jaccard_index(df_item_concept['Generation'], df_item_concept['Recognition']))


# %%
corr_mats = []
for i, model in enumerate(df_con.index.unique()):
    discrimination_model = discrimination.df[discrimination.df['model'] == model]
    generation_model = generation.df[generation.df['model'] == model]
    recognition_model = recognition.df[recognition.df['model'] == model]
    df_item_concept = pd.DataFrame({
        'Discrimination': discrimination_model['concept_response'],
        'Generation': generation_model['concept_response'],
        'Recognition': recognition_model['concept_response'],
    })
    corr_mats.append(df_item_concept[1:].corr())

# human
itemwise_df = human_df.groupby(['main_id', 'task'])['score'].mean().unstack() 
# correlation matrix
corr_mat = itemwise_df.corr()

human_df['shared'] = human_df['itemid'].apply(lambda x: str(x).split('_')[1] != '1')


human_df[human_df['shared'] == True].groupby(['participant_fk', 'task'])['score'].mean().unstack()

corr_mats_df = pd.concat(corr_mats)
corr_mats_df['model'] = np.repeat(df_con.index.unique(), 3)

mean_cor_mat = pd.concat(corr_mats).groupby(level=0).mean()
plt.figure(figsize=(8, 6))
plot_correlation_matrix(mean_cor_mat, vmin=-1, vmax=1)
plt.title('Average LLM', fontsize=20)


fig, ax = plt.subplots(4, 3, figsize=(20, 20))
for i, model in enumerate(df_con.index.unique()):
    plot_correlation_matrix(corr_mats[i], vmin=-1, vmax=1, ax=ax[i//3, i%3])
    ax[i//3, i%3].set_title(get_model_name(model), fontsize=25)

if i < 11:
    for j in range(i+1, 12):
        ax[j//3, j%3].axis('off')
plt.tight_layout()
plt.show()

