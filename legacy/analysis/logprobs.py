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
import seaborn as sns

mpl.rcParams['figure.dpi'] = 300


run_dir = 'run_2024-11-12' 
generation = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))
discrimination_example = Eval(os.path.join(RESULTS_DIR, 'run_example_2024-11-12', 'discrimination.json'))

#%% Plotting
mc_lp = discrimination.df.groupby('choice')['logprobs'].mean()
mc_lp_err = discrimination.df.groupby(['choice','model'])['logprobs'].mean().groupby('choice').sem()

abcd_lp = recognition.df.groupby('concept_response')['logprobs'].mean()[1]
abcd_lp_err = recognition.df.groupby(['concept_response','model'])['logprobs'].mean().groupby('concept_response').sem()

oe_concept_lp = generation.df.groupby('concept_response')['logprobs'].mean().loc[1]
oe_concept_lp_err = generation.df.groupby(['concept_response','model'])['logprobs'].mean().loc[1].sem()
or_matrix_lp = generation.df.groupby('matrix_response')['logprobs'].mean().loc[1]
or_matrix_lp_err = generation.df.groupby(['matrix_response','model'])['logprobs'].mean().loc[1].sem()
or_duplication_lp = generation.df.groupby('duplicate_response')['logprobs'].mean().loc[1]
or_duplication_lp_err = generation.df.groupby(['duplicate_response','model'])['logprobs'].mean().loc[1].sem()
oe_other_lp = generation.df.groupby('other_response')['logprobs'].mean().loc[1]
oe_other_lp_err = generation.df.groupby(['other_response','model'])['logprobs'].mean().loc[1].sem()

df_lp = pd.DataFrame({
    'Discrimination': mc_lp.to_dict(),
    'Generation': {'concept': oe_concept_lp,
                    'duplicate': or_duplication_lp,
                    'matrix': or_matrix_lp, 
                    'other': oe_other_lp},
    'Recognition': {'concept': abcd_lp,
                    'matrix': np.nan,
                    'other': abcd_lp},
}).T
df_lp_err = pd.DataFrame({
    'Discrimination': mc_lp_err,
    'Generation': {'concept': oe_concept_lp_err,
                   'duplicate': or_duplication_lp_err,
                   'matrix': or_matrix_lp_err, 
                   'other': oe_other_lp_err},
    'Recognition': {'concept': abcd_lp_err[1],
                    'matrix': np.nan,
                    'other': abcd_lp_err[0]},
}).T
df_lp = df_lp[['concept', 'matrix', 'duplicate', 'other']]

# Plot each x category separately in subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

df_lp.T['Generation'].dropna().plot(kind='bar', ax=axes[0], color=['#d44e86', '#665190', '#DEB841', '#adf4dc'], edgecolor='black', yerr=df_lp_err.T['Generation'])
axes[0].set_title('Generation', size=20)
axes[0].tick_params(axis='x', labelsize=18, rotation=0)
axes[0].set_ylabel('Confidence', size=20)
axes[0].set_ylim(0.9, 1)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
axes[0].tick_params(axis='y', labelsize=18)

df_lp.T['Discrimination'].dropna().plot(kind='bar', ax=axes[1], color=['#d44e86', '#665190', '#DEB841', '#adf4dc'], edgecolor='black', yerr=df_lp_err.T['Discrimination'])
axes[1].set_title('Discrimination', size=20)
axes[1].tick_params(axis='x', labelsize=18, rotation=0)
axes[1].set_ylim(0.4, 1)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
axes[1].tick_params(axis='y', labelsize=18)

df_lp.T['Recognition'].dropna().plot(kind='bar', ax=axes[2], color=['#d44e86', '#adf4dc'], edgecolor='black', yerr=df_lp_err.T['Recognition'].dropna())
axes[2].set_title('Recognition', size=20)
axes[2].tick_params(axis='x', labelsize=18, rotation=0)
axes[2].set_ylim(0.4, 1)
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
axes[2].tick_params(axis='y', labelsize=18)
 
# Adjust layout
# AC9969
# DEB841

plt.tight_layout()
plt.show()
# %%
discrimination_model = discrimination.df.groupby(['model','choice'])['logprobs'].mean().reset_index()
plt.figure(figsize=(10, 10))
sns.barplot(data=discrimination_model, x='model', y='logprobs', hue='choice')
plt.ylabel('Confidence')
plt.xlabel('Choice')
plt.title('Discrimination Logprobs')
plt.show()


# %%
# Calculate mean and SEM
recognition_model = recognition.df.groupby(['model', 'concept_response'])['logprobs'].mean().reset_index()
recognition_model_sem = recognition.df.groupby(['model', 'concept_response'])['logprobs'].sem().reset_index()

# Merge SEM values into the mean DataFrame for consistency
recognition_model = recognition_model.merge(
    recognition_model_sem.rename(columns={'logprobs': 'sem'}),
    on=['model', 'concept_response']
)

# Create the plot
plt.figure(figsize=(10, 10))
x = np.arange(len(recognition_model['model'].unique()))  # X positions
bar_width = 0.4

# Iterate over concept_response to plot each bar with error bars
for i, concept_response in enumerate(recognition_model['concept_response'].unique()):
    subset = recognition_model[recognition_model['concept_response'] == concept_response]
    plt.bar(
        x + i * bar_width - bar_width / 2,  # Adjust bar position
        subset['logprobs'],  # Heights
        bar_width,
        color= '#d44e86' if concept_response == 1 else '#adf4dc',  # Color
        edgecolor='black',  # Edge color
        yerr=subset['sem'],  # Error bars
        label = 'Concept' if concept_response == 1 else 'Other'
    )

# Customize axes
plt.ylabel('Confidence', size=16)
plt.xticks(
    ticks=x, 
    labels=[get_model_name(name) for name in recognition_model['model'].unique()],
    rotation=45,
    ha='right',
    size=14
)
plt.ylim(0, 1)
plt.yticks(size=14)
plt.title('Concept Recognition', size=20)
plt.legend(frameon=False, fontsize=14, bbox_to_anchor=(1, 1))

# Show plot
plt.tight_layout()
plt.show()

# %%
