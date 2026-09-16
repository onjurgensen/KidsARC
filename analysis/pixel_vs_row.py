#%% Imports
import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset
from utils.plot_utils import plot_item, get_percentage_ticks
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
from utils.human_preprocessing import df as human_df
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind
from rich import print as rprint
mpl.rcParams['figure.dpi'] = 300

above_below_items = ["0Zgi5T", "3qC5EW", "DAyZ1w", "MXPEB2", "NCz8AP", "TKKBpo", "ckjZ81", "hC2gHL", "zYuq0D"]
human_df = human_df[human_df['main_id'].isin(above_below_items)]

old_arc = False
if old_arc:
    RESULTS_DIR = RESULTS_DIR + '/old/2024-11-12'

def plot_pixel_vs_row(df_pixel, df_row, title=None, ylim=None):

    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.3  # Adjusted bar width for better spacing
    x = np.arange(len(df_pixel))

    # Plot bars with proper alignment
    ax.bar(x - bar_width / 2, df_pixel['matrix_response'], bar_width, color='#508FA3', edgecolor='black', label='Pixel Literal')
    ax.bar(x + bar_width / 2, df_row['matrix_response'], bar_width, color='#B53A3D', edgecolor='black', label='Row Literal')

    # Customize plot appearance
    ax.set_xticks(x)  # Align ticks with bar groups
    ax.set_xticklabels([get_model_name(i) for i in df_pixel.index], rotation=45, ha='right', fontsize=16)
    ax.set_ylabel('Literal Responses', fontsize=16)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(['{:.0f}%'.format(i * 100) for i in np.arange(0, 1.1, 0.2)], fontsize=16)
    ax.set_ylim(0, 1) if ylim is None else ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    plt.legend(frameon=False, fontsize=16)

    # Add title
    if title:
        plt.title(title, fontsize=18)

    plt.tight_layout()
    plt.show()

def compare_means(df_1: pd.DataFrame, df_2: pd.DataFrame):
    pixel_mean, pixel_std = df_1.groupby('model')['matrix_response'].mean().describe()[['mean', 'std']].values
    row_mean, row_std = df_2.groupby('model')['matrix_response'].mean().describe()[['mean', 'std']].values
    p_value = ttest_ind(df_1.groupby('model')['matrix_response'].mean(), df_2.groupby('model')['matrix_response'].mean()).pvalue
    return (pixel_mean, pixel_std), (row_mean, row_std), p_value


#%% Discrimination
file_pixel = os.path.join(RESULTS_DIR, 'pixel_vs_row/pixel.json')
file_row = os.path.join(RESULTS_DIR, 'pixel_vs_row/row.json')
file_generation = os.path.join(RESULTS_DIR, 'run_2024-11-12/generation.json')
pixel = Eval(file_pixel, old_arc=old_arc)
row = Eval(file_row, old_arc=old_arc)
pixel_reasoning = Eval(os.path.join(RESULTS_DIR, 'reasoning', 'pixel.json'), old_arc=old_arc)
row_reasoning = Eval(os.path.join(RESULTS_DIR, 'reasoning', 'row.json'), old_arc=old_arc)
 
pixel_df = pd.concat([pixel.df, pixel_reasoning.df]).reset_index(drop=True)
row_df = pd.concat([row.df, row_reasoning.df]).reset_index(drop=True)

# Means 
(pixel_mean, pixel_std), (row_mean, row_std), p_value = compare_means(pixel_df, row_df)
rprint(
    f"[u][i]Proportion Matrix Responses[/i][/u] (N = {len(pixel.df['model'].unique())})\n",
    '\nPixel:', 
    f'[yellow][u]{pixel_mean:.2f}[/u][/yellow]', f'({pixel_std:.2f})', 
    '\nRow:', 
    f'[yellow][u]{row_mean:.2f}[/u][/yellow]', f'({row_std:.2f})',
    '\n\nP-value:', 
    f'[green]{p_value:.3f}[/green]' if p_value < 0.05 else f'[red]{p_value:.3f}[/red]', 
)

human_discrimination = human_df[human_df['task'] == 'Discrimination']

# Plot the data
response_cols = ['concept_response', 'matrix_response']
df_row = row_df.groupby('model')[response_cols].mean()
df_row.loc['Human'] = human_discrimination['concept_answer'].mean(), human_discrimination['matrix_row_answer'].mean()
df_row.sort_values('matrix_response', ascending=True, inplace=True)

df_pixel = pixel_df.groupby('model')[response_cols].mean()
df_pixel.loc['Human'] = human_discrimination['concept_answer'].mean(), human_discrimination['matrix_pixel_answer'].mean()
df_pixel = df_pixel.loc[df_row.index]
plot_pixel_vs_row(df_pixel, df_row, 'Discrimination', ylim=(0, 0.6))


#%% Plot the items
# dataset_pixel = AmbigousARCDataset(d_matrix_level='pixel')
# dataset_row = AmbigousARCDataset(d_matrix_level='row')

# item_df = pixel.df.groupby('item_main_id')['matrix_response'].mean().reset_index()
# item_df.rename(columns={'matrix_response': 'matrix_response_pixel'}, inplace=True)
# item_df['matrix_response_row'] = row.df.groupby('item_main_id')['matrix_response'].mean().values
# item_df = item_df.sort_values('matrix_response_row', ascending=False).reset_index(drop=True)

# main_fig, main_axs = plt.subplots(len(item_df['item_main_id']), 2, figsize=(8, 30))
# for idx, item in enumerate(item_df['item_main_id']):
#     fig_pixel, _ = dataset_pixel.plot(item, return_fig=True)
#     fig_pixel.canvas.draw()
#     fig_row, _ = dataset_row.plot(item, return_fig=True)
#     fig_row.canvas.draw()

#     main_axs[idx, 0].imshow(np.asarray(fig_pixel.canvas.buffer_rgba()))
#     main_axs[idx, 0].set_title(f'Pixel - {item_df.loc[idx, "matrix_response_pixel"].round(2) * 100:.0f}%')
#     main_axs[idx, 1].imshow(np.asarray(fig_row.canvas.buffer_rgba()))
#     main_axs[idx, 1].set_title(f'Row - {item_df.loc[idx, "matrix_response_row"].round(2) * 100:.0f}%')
    
#     main_axs[idx, 0].axis('off')
#     main_axs[idx, 1].axis('off')
# plt.tight_layout()
# plt.show()


#%% Generation 
response_cols = ['concept_response', 'matrix_response', 'duplicate_response']
dataset_cfg = json.load(open(file_generation, 'rb'))['dataset_config']

# Pixel
dataset_cfg['d_matrix_level'] = 'pixel'
dataset = AmbigousARCDataset(**dataset_cfg)
generation_pixel = Eval(file_generation, dataset=dataset, old_arc=old_arc)
gen_pixel_df = generation_pixel.df[generation_pixel.df['item_main_id'].isin(above_below_items)]
assert gen_pixel_df.shape[0] == pixel.df.shape[0], 'Different number of items in the generation and discrimination datasets'

# Row
dataset_cfg['d_matrix_level'] = 'row'
dataset = AmbigousARCDataset(**dataset_cfg)
generation_row = Eval(file_generation, dataset=dataset, old_arc=old_arc)
gen_row_df = generation_row.df[generation_row.df['item_main_id'].isin(above_below_items)]
assert gen_row_df.shape[0] == row.df.shape[0], 'Different number of items in the generation and discrimination datasets'

# Means
(pixel_mean, pixel_std), (row_mean, row_std), p_value = compare_means(gen_pixel_df, gen_row_df)
rprint(
    f"[u][i]Proportion Matrix Responses[/i][/u] (N = {len(gen_pixel_df['model'].unique())})\n",
    '\nPixel:', 
    f'[yellow][u]{pixel_mean:.2f}[/u][/yellow]', f'({pixel_std:.2f})', 
    '\nRow:', 
    f'[yellow][u]{row_mean:.2f}[/u][/yellow]', f'({row_std:.2f})',
    '\n\nP-value:', 
    f'[green]{p_value:.3f}[/green]' if p_value < 0.05 else f'[red]{p_value:.3f}[/red]', 
)

# Human
human_generation = human_df[human_df['task'] == 'Generation']

# Plot the data
response_cols = ['concept_response', 'matrix_response']
df_row = gen_row_df.groupby('model')[response_cols].mean()
df_row.loc['Human'] = human_generation['concept_answer'].mean(), human_generation['matrix_row_answer'].mean()
df_row.sort_values('matrix_response', ascending=True, inplace=True)

df_pixel = gen_pixel_df.groupby('model')[response_cols].mean()
df_pixel.loc['Human'] = human_generation['concept_answer'].mean(), human_generation['matrix_pixel_answer'].mean()
df_pixel = df_pixel.loc[df_row.index]
plot_pixel_vs_row(df_pixel, df_row, 'Generation', ylim=(0, 0.4))

#%% Rotated items

for response in ['matrix_response']:

    # Original
    row_ms = [row.df[response].mean()]
    row_model_ms = [row.df.groupby('model')[response].mean().sort_index().values]
    row_sems = [row.df.groupby('model')[response].mean().sem()]
    pixel_ms = [pixel.df[response].mean()]
    pixel_model_ms = [pixel.df.groupby('model')[response].mean().sort_index().values]
    pixel_sems = [pixel.df.groupby('model')[response].mean().sem()]

    # Rotated
    for angle in [90, 180, 270]:
        file_row_rotated = os.path.join(RESULTS_DIR, f'pixel_vs_row/row_rotated_{angle}.json')
        file_pixel_rotated = os.path.join(RESULTS_DIR, f'pixel_vs_row/pixel_rotated_{angle}.json')
        row_rotated = Eval(file_row_rotated, old_arc=old_arc)
        pixel_rotated = Eval(file_pixel_rotated, old_arc=old_arc)

        row_ms.append(row_rotated.df[response].mean())
        row_sems.append(row_rotated.df.groupby('model')[response].mean().sem())
        row_model_ms.append(row_rotated.df.groupby('model')[response].mean().sort_index().values)
        pixel_ms.append(pixel_rotated.df[response].mean())
        pixel_model_ms.append(pixel_rotated.df.groupby('model')[response].mean().sort_index().values)
        pixel_sems.append(pixel_rotated.df.groupby('model')[response].mean().sem())

    plt.figure(figsize=(4.5, 5))

    # Extract x-axis values (rotations)
    x_values = [0, 90, 180, 270]
    alpha = 0.25
    linewidth = 1.3

    # Plot individual model lines with low opacity
    for model_idx in range(11):  # Iterate over each model (11 models)
        row_values = [row_model_ms[rot][model_idx] for rot in range(4)]
        pixel_values = [pixel_model_ms[rot][model_idx] for rot in range(4)]

        plt.plot(x_values, row_values, color='#B53A3D', alpha=alpha, linewidth=linewidth)  # Row model lines
        #plt.plot(x_values, pixel_values, color='#508FA3', alpha=alpha, linewidth=linewidth)  # Pixel model lines

    # Plot means with error bars
    plt.errorbar([0, 90, 180, 270], row_ms, yerr=row_sems, label='LLMs', color='#B53A3D', marker='o')
    #plt.errorbar([0, 90, 180, 270], pixel_ms, yerr=pixel_sems, label='Pixel Literal', color='#508FA3', marker='o')
    plt.axhline(y=0.005, color='#508FA3', linestyle='--', linewidth=linewidth, label='Human')

    # Additional plot styling
    plt.xticks([0, 90, 180, 270], ['0°', '90°', '180°', '270°'], fontsize=12)
    plt.xlabel('Rotation', fontsize=12)
    plt.title(f'{response.split('_')[0].capitalize()} Response', fontsize=14)
    plt.yticks(np.linspace(0, 0.6, num=7), [f'{x:.0%}' for x in np.linspace(0, 0.6, num=7)], size=12)
    plt.ylim(0, 0.6)
    # white background of the legend
    plt.legend(loc='upper right', edgecolor='white')
    plt.tight_layout()
    plt.show()

#%% Plot the items
dataset_row_original = AmbigousARCDataset(d_matrix_level='row', matrix_rotation=0)
dataset_row_rotated = AmbigousARCDataset(d_matrix_level='row', matrix_rotation=90)

item_df = row.df.groupby('item_main_id')['matrix_response'].mean().reset_index()
item_df.rename(columns={'matrix_response': 'matrix_response_row_original'}, inplace=True)
item_df['matrix_response_row_rotated'] = row_rotated.df.groupby('item_main_id')['matrix_response'].mean().values
item_df = item_df.sort_values(['matrix_response_row_rotated', 'matrix_response_row_original'], ascending=False).reset_index(drop=True)

main_fig, main_axs = plt.subplots(len(item_df['item_main_id']), 2, figsize=(8, 30))
for idx, item in enumerate(item_df['item_main_id']):
    fig_pixel, _ = dataset_row_original.plot(item, return_fig=True)
    fig_pixel.canvas.draw()
    fig_row, _ = dataset_row_rotated.plot(item, return_fig=True)
    fig_row.canvas.draw()

    main_axs[idx, 0].imshow(np.asarray(fig_pixel.canvas.buffer_rgba()))
    main_axs[idx, 0].set_title(f'Original - {item_df.loc[idx, "matrix_response_row_original"].round(2) * 100:.0f}%')
    main_axs[idx, 1].imshow(np.asarray(fig_row.canvas.buffer_rgba()))
    main_axs[idx, 1].set_title(f'Rotated - {item_df.loc[idx, "matrix_response_row_rotated"].round(2) * 100:.0f}%')
    
    main_axs[idx, 0].axis('off')
    main_axs[idx, 1].axis('off')
plt.tight_layout()
plt.show()

# %%
print(dataset_row_original.x[0])
dataset_row_original.plot(0)
print(dataset_row_rotated.x[0])

dataset_row_rotated.plot(0)

# %%
def plot_item_responses(item_id):
    row.df[row.df['item_main_id'] == item_id].groupby('model')['concept_response'].mean().plot(kind='bar', figsize=(10, 5))
plot_item_responses('MXPEB2')
plot_item_responses('ckjZ81')

row.dataset.plot('MXPEB2', show_mirrors=True)
row.dataset.plot('ckjZ81', show_mirrors=True)

mean = row.df[row.df['item_main_id'] == 'MXPEB2']['matrix_response'].mean().round(2)
AmbigousARCDataset(task='discrimination', d_matrix_level='row', canonicize=True).plot('MXPEB2', title=f'MXPEB2 || Matrix: {mean * 100:.0f}%')
mean = row.df[row.df['item_main_id'] == 'ckjZ81']['matrix_response'].mean()
AmbigousARCDataset(task='discrimination', d_matrix_level='row', canonicize=True).plot('ckjZ81', title=f'ckjZ81 || Matrix: {mean * 100:.0f}%')

# %%

angle = 270
AmbigousARCDataset(task='generation', d_matrix_level='row', matrix_rotation=angle).plot('3qC5EW')
row_rotated = Eval(os.path.join(RESULTS_DIR, f'pixel_vs_row/example/row_rotated_{angle}.json'))
print(row_rotated.dataset.x[8])
row.dataset.plot(8)

row.dataset.x[8]

angle = 90
file_row_rotated = os.path.join(RESULTS_DIR, f'pixel_vs_row/example/row_rotated_{angle}.json')
file_pixel_rotated = os.path.join(RESULTS_DIR, f'pixel_vs_row/example/pixel_rotated_{angle}.json')

pixel_rotated = Eval(file_pixel_rotated)
# %%



row.dataset.plot('3qC5EW')