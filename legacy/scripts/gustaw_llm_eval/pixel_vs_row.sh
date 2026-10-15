#!/bin/bash

# Load models from YAML file
yaml() { python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('$1'))$2))" ; }
models=($(yaml globals.yml "['MODEL_IDS']"))

# Set items
items=(
    '0Zgi5T'
    '3qC5EW'
    'DAyZ1w'
    'MXPEB2'
    'NCz8AP'
    'TKKBpo'
    'ckjZ81'
    'hC2gHL'
    'zYuq0D'
)

printf "Running pixel...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --random_mat \
    --find_best_seed \
    --duplicate c \
    --n_mirror 4 \
    --example_item different \
    --encoding int \
    --d_matrix_level pixel \
    --filter_items_list "${items[@]}" \
    --tasks discrimination \
    --results_dir data/results/pixel_vs_row \
    --file_name pixel

# Get best seed (the tasks are the same from now on)
best_seed=$(jq -r '.dataset_config.seed' data/results/pixel_vs_row/pixel.json)

printf "Running row...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --random_mat \
    --no-find_best_seed \
    --seed $best_seed \
    --duplicate c \
    --n_mirror 4 \
    --example_item different \
    --encoding int \
    --d_matrix_level row \
    --filter_items_list "${items[@]}" \
    --tasks discrimination \
    --results_dir data/results/pixel_vs_row \
    --file_name row

for i in 90 180 270; do
    printf "Running rotated ${i}...\n"

    python3 query_models.py \
        --models "${models[@]}" \
        --random_mat \
        --no-find_best_seed \
        --seed $best_seed \
        --duplicate c \
        --n_mirror 4 \
        --example_item different \
        --encoding int \
        --d_matrix_level pixel \
        --matrix_rotation $i \
        --filter_items_list "${items[@]}" \
        --tasks discrimination \
        --results_dir data/results/pixel_vs_row \
        --file_name pixel_rotated_${i}

    python3 query_models.py \
        --models "${models[@]}" \
        --random_mat \
        --no-find_best_seed \
        --seed $best_seed \
        --duplicate c \
        --n_mirror 4 \
        --example_item different \
        --encoding int \
        --d_matrix_level row \
        --matrix_rotation $i \
        --filter_items_list "${items[@]}" \
        --tasks discrimination \
        --results_dir data/results/pixel_vs_row \
        --file_name row_rotated_${i}
done