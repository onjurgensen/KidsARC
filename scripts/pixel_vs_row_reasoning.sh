#!/bin/bash

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

while true; do
    printf "Running pixel...\n"
    python3 query_models.py \
        --models deepseek-reasoner \
        --random_mat \
        --find_best_seed \
        --duplicate c \
        --n_mirror 4 \
        --example_item different \
        --encoding int \
        --d_matrix_level pixel \
        --query_timeout 1200 \
        --filter_items_list "${items[@]}" \
        --tasks discrimination \
        --results_dir data/results/reasoning \
        --file_name pixel \
        --intermediate_results 
    
    if [ $? -eq 0 ]; then
        echo "Script finished successfully."
        break
    else
        echo "Script failed. Retrying in 10 seconds..."
        sleep 10
    fi
done

# Get best seed (the tasks are the same from now on)
best_seed=$(jq -r '.dataset_config.seed' data/results/reasoning/pixel.json)

while true; do
    printf "Running row...\n"
    python3 query_models.py \
        --models deepseek-reasoner \
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
        --results_dir data/results/reasoning \
        --file_name row \
        --intermediate_results 
    
    if [ $? -eq 0 ]; then
        echo "Script finished successfully."
        break
    else
        echo "Script failed. Retrying in 10 seconds..."
        sleep 10
    fi
done


for i in 90 180 270; do

    while true; do
        printf "Running rotated pixel ${i}...\n"
        python3 query_models.py \
            --models deepseek-reasoner \
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
            --results_dir data/results/reasoning \
            --file_name pixel_rotated_${i} \
            --intermediate_results 
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully."
            break
        else
            echo "Script failed. Retrying in 10 seconds..."
            sleep 10
        fi
    done

    while true; do
        printf "Running rotated row ${i}...\n"
        python3 query_models.py \
            --models deepseek-reasoner \
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
            --results_dir data/results/reasoning \
            --file_name row_rotated_${i} \
            --intermediate_results 
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully."
            break
        else
            echo "Script failed. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done