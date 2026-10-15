import os
from utils.globals import RESULTS_DIR
from utils.prompt_utils import AmbigousARCDataset
import json

seed = json.load(open(os.path.join(RESULTS_DIR, 'pixel_vs_row/row.json')))['dataset_config']['seed']
dataset = AmbigousARCDataset(task='discrimination', d_matrix_level='row', canonicize=True, seed=seed)

# Get the indices of the matrices
mat_id = 'MXPEB2_1'
con_id = 'ckjZ81_1'
mat_idx = [idx for idx, item in enumerate(dataset.items_data) if item['id'] == mat_id][0]
con_idx = [idx for idx, item in enumerate(dataset.items_data) if item['id'] == con_id][0]

print(dataset.x[con_idx])

dataset.plot(mat_idx)
dataset.plot(dataset.example_items[mat_idx])