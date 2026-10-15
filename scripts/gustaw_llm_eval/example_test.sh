#!/bin/bash

# Load models from YAML file
# yaml() { python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('$1'))$2))" ; }
# models=($(yaml globals.yml "['MODEL_IDS']"))
models=(
    mistralai/Mixtral-8x7B-Instruct-v0.1
    Qwen/Qwen2.5-7B-Instruct-Turbo
    Qwen/Qwen2.5-72B-Instruct-Turbo
    meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
    meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
    meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
)

printf "Running No Example...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --random_mat \
    --find_best_seed \
    --duplicate c \
    --n_mirror 1 \
    --example_item no_example \
    --encoding int \
    --d_matrix_level pixel \
    --results_dir data/results/example_test \
    --file_names no_example

printf "Running Example Other Concept...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --random_mat \
    --find_best_seed \
    --duplicate c \
    --n_mirror 1 \
    --example_item different \
    --encoding int \
    --d_matrix_level pixel \
    --results_dir data/results/example_test \
    --file_names different_concept

printf "Running Example Same Concept...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --random_mat \
    --find_best_seed \
    --duplicate c \
    --n_mirror 1 \
    --example_item same \
    --encoding int \
    --d_matrix_level pixel \
    --results_dir data/results/example_test \
    --file_names same_concept