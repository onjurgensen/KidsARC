#!/bin/bash

while true; do
    python3 query_models.py \
        --models deepseek-reasoner \
        --concept_answer_n 4 \
        --random_mat \
        --find_best_seed \
        --duplicate c \
        --n_mirror 0 \
        --example_item different \
        --encoding int \
        --mirror_type color \
        --query_timeout 1200 \
        --results_dir data/results/reasoning \
        --intermediate_results 
    
    if [ $? -eq 0 ]; then
        echo "Script finished successfully."
        break
    else
        echo "Script failed. Retrying in 10 seconds..."
        sleep 10
    fi
done