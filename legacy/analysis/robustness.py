import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.plot_utils import plot_item
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, MODEL_NAMES, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

run_dir = 'run_2024-11-12'
generation_color = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination_color = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition_color = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

run_dir = 'run_example_2024-11-12'
generation_example = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination_example = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition_example = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

run_dir = 'run_answeropt_2024-11-12'
discrimination_answeropt = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))


robustness_df = pd.DataFrame({
    'model': discrimination_answeropt.models_names,
    'color': np.nan,
    'example': np.nan,
    'answeropt': np.nan
})

for mirror in ['color', 'example', 'answeropt']:
    mirror_eval = eval(f"discrimination_{mirror}")
    robustness_df[mirror] = mirror_eval.robustness
    
    print(
        f'{mirror}\n{np.mean(mirror_eval.robustness).round(2)}\n'
    )

robustness_df = robustness_df.sort_values(by=['color', 'example', 'answeropt'], ascending=True)

plt.figure(figsize=(10, 6))
width = 0.25  # Width of each bar
x = range(len(MODEL_NAMES))  # X positions for the bars

# Plotting each category
plt.bar(x, robustness_df['color'], width, color='blue', alpha=0.5, label='Color')
plt.bar([i + width for i in x], robustness_df['example'], width, color='grey', alpha=0.5, label='Example')
plt.bar([i + 2 * width for i in x], robustness_df['answeropt'], width, color='green', alpha=0.5, label='AnswerOpt')

# Customizing plot
plt.xticks([i + width for i in x], [get_model_name(model) for model in robustness_df['model']], rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Robustness', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()


# Combine the dataframes
df = pd.DataFrame({})
for task in ['generation', 'discrimination', 'recognition']:
    # Create DataFrame
    df_color = pd.DataFrame(eval(f'{task}_color').df.groupby(['model', 'mirror'])['concept_response'].mean().reset_index())
    df_example = pd.DataFrame(eval(f'{task}_example').df.groupby(['model', 'mirror'])['concept_response'].mean().reset_index())
    df_color['example_acc'] = df_example['concept_response']
    df_task = df_color.rename(columns={'concept_response': 'color_acc'})
    df_task['task'] = task
    df_task['model'] = df_task['model'].apply(lambda x: get_model_name(x))
    df = pd.concat([df, df_task])

# Plotting
offset = 0.13
for task in ['generation', 'discrimination', 'recognition']:
    df_task = df[df['task'] == task]
    means = df_task.groupby('model')[['example_acc']].mean().sort_values('example_acc', ascending=True)
    df_task['model'] = pd.Categorical(df_task['model'], categories=means.index, ordered=True)
    df_task = df_task.sort_values('model')

    plt.figure(figsize=(8, 6))
    for i, (model, group) in enumerate(df_task.groupby('model')):
        # Scatter plot for 'example_acc' with a slight offset to the left
        plt.scatter(
            [i - offset] * len(group),
            group['example_acc'],
            label='example_acc',
            s=100,
            alpha=0.5,
            color='grey'
        )
        
        # Scatter plot for 'color_acc' with a slight offset to the right
        plt.scatter(
            [i + offset] * len(group),
            group['color_acc'],
            label='color_acc',
            s=100,
            alpha=0.5,
            color='blue'
        )
    plt.xticks([i for i in range(len(MODEL_NAMES))], [get_model_name(i) for i in df_task.groupby('model')['model'].first().values], rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Concept Response', fontsize=16)
    plt.tight_layout()
    plt.title(f'{task.capitalize()} Task', fontsize=20)
    #plt.legend()
    plt.show()