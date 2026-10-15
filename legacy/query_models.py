import os
import json
from datetime import date
import warnings
from typing import *
import argparse
from utils.globals import ITEMS_FILE, RESULTS_DIR
from utils.prompt_utils import AmbigousARCDataset
from utils.query_utils import run_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the multiple choice experiment')

    parser.add_argument('--items_file', type=str, default=ITEMS_FILE, help='Path to the items file')
    parser.add_argument('--models', nargs='+', default=[], type=str, help='Models to run the experiment on')
    parser.add_argument('--random_mat', type=bool, help='Randomize the matrix', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Random seed')
    parser.add_argument('--find_best_seed', type=bool, default=False, help='Find the best seed', action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_iter_seed', type=int, default=100, help='Number of iterations to find the best seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--n_mirror', type=int, default=0, help='Number of mirror items')
    parser.add_argument('--mirror_type', type=str, default='color', help='Mirror type (color or example)')
    parser.add_argument('--concept_answer_n', type=int, default=4, help='Number of concept answers')
    parser.add_argument('--duplicate', type=str, default=None, help='Duplicate type')
    parser.add_argument('--encoding', type=str, default='int', help='Encoding type (int or color)')
    parser.add_argument('--d_matrix_level', type=str, default='pixel', help='D matrix level (pixel or row)')
    parser.add_argument('--matrix_rotation', type=int, default=0, help='Matrix rotation (0, 90, 180, 270)')
    parser.add_argument('--results_dir', type=str, help='Results directory')
    parser.add_argument('--example_item', type=str, default='different', help='Example item type (same, different or None)')
    parser.add_argument('--tasks', nargs='+', default=['generation', 'discrimination', 'recognition'], type=str, help='Tasks to run the experiment on')
    parser.add_argument('--filter_items_list', nargs='+', type=str, help='Item IDs to run the experiment on', default=None)
    parser.add_argument('--logprobs', type=bool, default=True, help='Logprobs', action=argparse.BooleanOptionalAction)
    parser.add_argument('--file_names', nargs='+', type=str, help='File names', default=None)
    parser.add_argument('--intermediate_results', type=bool, default=False, help='Save intermediate results', action=argparse.BooleanOptionalAction)
    parser.add_argument('--query_timeout', type=int, default=60, help='Query timeout')

    args = parser.parse_args()


    if args.seeds and len(args.seeds) != len(args.tasks):
        raise ValueError('The number of seeds should be equal to the number of tasks')

    if args.seeds and args.find_best_seed:
        warnings.warn('Both seeds and find_best_seed are provided. The seeds will be ignored.')    
    
    elif not args.seeds and not args.find_best_seed:
        warnings.warn('No seed is provided. The results will be non-deterministic.')

    if args.concept_answer_n != (0 if args.duplicate is None else 1 + args.random_mat + 2):
        warnings.warn('The number of multiple choice answer options in recognition and discrimination tasks is not equal.')

    if args.file_names:
        if len(args.file_names) == 1 and len(args.tasks) > 1:
            warnings.warn('The number of file names is not equal to the number of tasks')
            args.file_names = [f'{task}_{args.file_names[0]}' for task in args.tasks]
        
        if len(set(args.file_names)) != len(args.file_names):
            raise ValueError('The file names should be unique')
    

    for idx, task in enumerate(args.tasks):
        print(f'\n{'-_'*5}Running task: {task}{'_-'*5}\n')

        # Set the results directory
        results_dir = args.results_dir if args.results_dir else os.path.join(RESULTS_DIR, f'run_{date.today()}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Set the file path
        file_name = args.file_names[idx] if args.file_names else task
        file_path = os.path.join(results_dir, f'{file_name}.json')

        # Create the dataset
        print('Creating the dataset...')
        dataset = AmbigousARCDataset(
            task = task,
            items_data = args.items_file,
            batch_size = args.batch_size,
            random_mat = args.random_mat,
            example_item = args.example_item,
            n_mirror = args.n_mirror,
            concept_answer_n = args.concept_answer_n,
            duplicate = args.duplicate,
            encoding = args.encoding,
            mirror_type = args.mirror_type,
            d_matrix_level = args.d_matrix_level,
            matrix_rotation = args.matrix_rotation,
            filter_items_list = args.filter_items_list,
            seed = args.seeds[idx]
                if args.seeds 
                else None,
            find_best_seed = args.find_best_seed 
                if task != 'generation'
                else False,
            n_iter_seed = args.n_iter_seed
        )

        # Run the experiment
        print('Running the experiment...')
        results = run_experiment(
            input_prompts = dataset.x, 
            logprobs = args.logprobs,
            models = args.models,
            system_prompt = "What is the concept that best describes the following task? Only give the answer, no other words or text."
                if task == 'recognition' 
                else "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.",
            max_tokens = 100 
                if task == 'generation' 
                else 5,
            query_timeout = args.query_timeout,
            results_path = file_path,
            intermediate_path = os.path.join(results_dir, f'{file_name}_temp.json') if args.intermediate_results else None
        )

        # Save the dataset configuration
        results['dataset_config'] = dataset.get_config()
        
        # Save the results
        json.dump(results, open(file_path, 'w'), indent=4)