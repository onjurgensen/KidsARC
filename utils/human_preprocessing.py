import json
import pandas as pd

# def print_full(x: pd.DataFrame | pd.Series):
#     s = x.to_markdown()
#     return s

def get_item_by_id(item_id: str):
    for item in items:
        if item['item_id'] == item_id:
            return item
    print(f'Item not found: {item_id}')
    return None

def item_left_unanswered(response):
    return 1 if all(e == '0' for e in list(response)) else 0

def score_generation(response, item, response_type='Concept'):
    return 1 if response == item[f'D_{response_type}'] else 0

def score_discrimination(response, response_type='Concept'):
    return 1 if response == f'D_{response_type}' else 0

def score_recognition(response, item):
    return 1 if response == item['concept'] else 0

def score(row, response, response_type='Concept'):
    item = get_item_by_id(row['itemid'])
    if row['task'] == 'Generation':
        return score_generation(response, item, response_type)
    
    elif row['task'] == 'Discrimination':
        return score_discrimination(response, response_type)
    
    elif row['task'] == 'Recognition':
        if response_type == 'Concept':
            return score_recognition(response, item)
    return 0

# Load data
items = json.load(open('data/human_data_collection/AmbigousARC.json', 'rb'))
df = pd.read_csv('data/human_data_collection/human_data.csv')
df_user = pd.read_csv('data/human_data_collection/user_data.csv')

# Capitalize tasks
df['task'] = df['task'].str.capitalize()

# Main ID column (as first column)
df['main_id'] = df['itemid'].apply(lambda x: str(x).split('_')[0])
df = df[['main_id'] + [col for col in df.columns if col != 'main_id']]

# Convert timestamps to datetime
df_user['created_timestamp'] = pd.to_datetime(df_user['created_timestamp']) 
df['saveitem_timestamp'] = pd.to_datetime(df['saveitem_timestamp'])

# Filter df_user to only include participants that have consented, were created between 2025-01-19 and 2025-01-21 and are from Prolific
df_user = df_user[
    (df_user['created_timestamp'] > '2025-01-19') &
    (df_user['created_timestamp'] < '2025-01-21') &
    (df_user.apply(lambda x: x['participantnr'].startswith('prolific'), axis=1))
]
print('Initial number of participants:', len(df_user))
print(f'Participants who did not consent: {len(df_user[df_user["consent"] == 0])}')
df_user = df_user[df_user['consent'] == 1]

# Filter df to only include participants present in the filtered df_user
df = df[df['participant_fk'].isin(df_user['user_id'])]

# Keep only participants with more than 70 tasks
task_counts = df.groupby('participant_fk')['task'].size()
participants_with_more_tasks = task_counts[task_counts > 70].index
df = df[df['participant_fk'].isin(participants_with_more_tasks)]
print(f'Participants with less than 70 tasks: {len(task_counts[task_counts < 70])} ({len(task_counts[task_counts < 70]) / len(df["participant_fk"].unique()) * 100:.0f}%)')

# Calculate the score for each task
df['concept_answer'] = df.apply(lambda x: score(x, x['response'], 'Concept'), axis=1)
df['matrix_pixel_answer'] = df.apply(lambda x: score(x, x['response'], 'Matrix'), axis=1)
df['matrix_row_answer'] = df.apply(lambda x: score(x, x['response'], 'Matrix_Row'), axis=1)

# Keep only participants that did not leave 80% of the generation tasks unanswered
generation_tasks = df[df['task'] == 'Generation']
unanswered_ratio = generation_tasks.groupby('participant_fk')['response'].apply(
    lambda x: sum(
        [item_left_unanswered(item) for item in x]
    ) / len(x)
)
participants_with_unanswered_generation = unanswered_ratio[unanswered_ratio > 0.8].index
df = df[~df['participant_fk'].isin(participants_with_unanswered_generation)]
print(f'Participants with > 80% unanswered generation tasks: {len(participants_with_unanswered_generation)} ({len(participants_with_unanswered_generation) / len(df["participant_fk"].unique()) * 100:.0f}%)')

# Keep only participants with at least 1 correct generation task
correct_sum = df[df['task'] == 'Generation'].groupby('participant_fk')['concept_answer'].sum()
participants_with_correct_generation = correct_sum[correct_sum > 0].index
df = df[df['participant_fk'].isin(participants_with_correct_generation)]
print(f'Participants that got less than 1 generation item correct: {len(correct_sum[correct_sum < 1])} ({len(correct_sum[correct_sum < 1]) / len(df["participant_fk"].unique()) * 100:.0f}%)')

# Keep only participants above chance level in discrimination and recognition
df_discrimination = df[df['task'] == 'Discrimination']
df_recognition = df[df['task'] == 'Recognition']
df_discrimination = df_discrimination.groupby('participant_fk')['concept_answer'].sum() / df_discrimination.groupby('participant_fk').size()
df_recognition = df_recognition.groupby('participant_fk')['concept_answer'].sum() / df_recognition.groupby('participant_fk').size()

participants_above_chance_discrimination = df_discrimination[df_discrimination > 0.25].index
participants_above_chance_recognition = df_recognition[df_recognition > 0.25].index

df = df[df['participant_fk'].isin(participants_above_chance_discrimination) & df['participant_fk'].isin(participants_above_chance_recognition)]
print(f'Participants below chance level in multiple choice tasks: {len(
    set(
        df_discrimination[df_discrimination < 0.25].index.to_list() + 
        df_recognition[df_recognition < 0.25].index.to_list()
        )
    )} ({len(
        set(
            df_discrimination[df_discrimination < 0.25].index.to_list() + 
            df_recognition[df_recognition < 0.25].index.to_list()
            )
        ) / len(df["participant_fk"].unique()) * 100:.0f}%)'
)

# Keep only participants with no more than 20% very fast responses
df['rt'] = df['rt'] / 1000
df['fast_response'] = df['rt'] < 2
fast_response_ratio = df.groupby('participant_fk')['fast_response'].mean()
participants_with_fast_responses = fast_response_ratio[fast_response_ratio > 0.2].index
df = df[~df['participant_fk'].isin(participants_with_fast_responses)]
print(f'Participants with > 20% very fast responses (< 2s): {len(participants_with_fast_responses)} ({len(participants_with_fast_responses) / len(df["participant_fk"].unique()) * 100:.0f}%)')

# Keep only participants with no more than 20% very slow responses
print(f'Final number of participants: {len(df["participant_fk"].unique())}')
