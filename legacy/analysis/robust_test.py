import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset, AmbigousARCDataset_old
from utils.plot_utils import plot_item
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

old_run_dir = 'old/2024-11-12/run_2024-11-12' 
new_run_dir = 'run_2024-11-12'

# Old 
generation = Eval(os.path.join(RESULTS_DIR, old_run_dir, 'generation.json'), old_arc=True)
recognition = Eval(os.path.join(RESULTS_DIR, old_run_dir, 'recognition.json'), old_arc=True)
discrimination = Eval(os.path.join(RESULTS_DIR, old_run_dir, 'discrimination.json'), old_arc=True)
discrimination_example = Eval(os.path.join(RESULTS_DIR, 'run_example_2024-11-12', 'discrimination.json'), old_arc=True)

# New 
discrimination_new = Eval(os.path.join(RESULTS_DIR, new_run_dir, 'discrimination.json'), old_arc=False)
discrimination_example_new = Eval(os.path.join(RESULTS_DIR, new_run_dir, 'discrimination.json'), old_arc=False)


item_df_color = discrimination.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df_example = discrimination_example.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df_all = pd.concat([item_df_color, item_df_example]).groupby(level=0).mean()
item_df_all.sort_values('matrix_response', ascending=False).head(10)

item_df_color_new = discrimination_new.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df_example_new = discrimination_example_new.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df_all_new = pd.concat([item_df_color_new, item_df_example_new]).groupby(level=0).mean()
item_df_all_new.sort_values('matrix_response', ascending=False).head(10)


response_type = 'concept_response'
np.corrcoef(item_df_all[response_type], item_df_all_new[response_type])[0, 1]
plt.scatter(item_df_all[response_type], item_df_all_new[response_type])
# diagonal line
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.ylabel('Prop Matrix Response')
plt.xlabel('Old')

discrimination.dataset.item_ids.index('zjvdIo_1')

discrimination.dataset.plot('zjvdIo_1')

item_df_all
