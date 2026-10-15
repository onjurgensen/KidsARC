import yaml
import os

def get_model_name(model_id):
    return MODEL_DICT.get(model_id, model_id)

# Define the path to the globals file
base_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(base_dir, '..'))
globals_path = os.path.join(repo_dir, 'globals.yml')

# Load data from YAML
with open(globals_path, "r") as stream:
    data = yaml.safe_load(stream)

def _path(key):
    """Absolute path for a FILE/DIR key (given relative to the repo root)."""
    return os.path.abspath(os.path.join(repo_dir, data[key]))

# Data paths
ITEMS_FILE = _path('ITEMS_FILE')
ITEMS_EXP_FILE = _path('ITEMS_EXP_FILE')
IMPORTS_DIR = _path('IMPORTS_DIR')
KEYS_FILE = _path('KEYS_FILE')
LLM_RESULTS_DIR = _path('LLM_RESULTS_DIR')
HUMAN_RESULTS_DIR = _path('HUMAN_RESULTS_DIR')
OUTPUT_DIR = _path('OUTPUT_DIR')

# Legacy name used by the collaborator's analysis code
RESULTS_DIR = LLM_RESULTS_DIR

# Model display names (legacy; only set in globals_old.yml)
MODEL_IDS = data.get('MODEL_IDS', [])
MODEL_NAMES = data.get('MODEL_NAMES', [])
MODEL_DICT = {id: name for id, name in zip(MODEL_IDS, MODEL_NAMES)}
