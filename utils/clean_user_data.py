"""
Merge the registered-user exports into one participant table.

    python -m utils.clean_user_data

Reads IMPORTS_DIR/user_kidsarc.csv and IMPORTS_DIR/user.csv, and writes
HUMAN_RESULTS_DIR/user_data_merged.csv (paths from globals.yml).
"""
import os

import pandas as pd

from utils.globals import IMPORTS_DIR, HUMAN_RESULTS_DIR

ARC_USERS_FILE = os.path.join(IMPORTS_DIR, 'user_kidsarc.csv')
FULL_USERS_FILE = os.path.join(IMPORTS_DIR, 'user.csv')
OUT_FILE = os.path.join(HUMAN_RESULTS_DIR, 'user_data_merged.csv')


def age_mapper(full_rows, arc_rows):
    # Prefer a directly recorded age, first in full_user_df, then arc_user_df.
    for rows in (full_rows, arc_rows):
        if rows.empty:
            continue

        row = rows.iloc[0]
        age = row.get("age")
        dob = row.get("dob")
        timestamp = row.get("created_timestamp")

        if pd.notna(age):
            return age

        if pd.notna(dob) and pd.notna(timestamp):
            completion_year = pd.to_datetime(timestamp).year
            return completion_year - int(dob)

    return None


def merge_users(arc_user_df, full_user_df):
    arc_user_df = arc_user_df.copy()
    full_user_df = full_user_df.copy()
    arc_user_df["participantnr"] = arc_user_df["participantnr"].astype("string")
    full_user_df["user_id"] = full_user_df["user_id"].astype("string")

    data_dict = {
        "type": [],
        "participant_id": [],
        "participant_id_arc": [],
        "research_study": [],
        "age": [],
        "language": [],
    }

    for participantnr in arc_user_df["participantnr"].dropna().unique():
        arc_rows = arc_user_df[arc_user_df["participantnr"] == participantnr]
        full_rows = full_user_df[full_user_df["user_id"] == participantnr]

        arc_user_id = arc_rows["user_id"].iloc[0]

        if not full_rows.empty:
            participant_type = "event"
            participant_id = participantnr
            research_study = full_rows["research_study"].iloc[0]
            age = age_mapper(full_rows, arc_rows)
            language = full_rows["language"].iloc[0]

        elif "test" in participantnr:
            participant_type = "test"
            participant_id = None
            research_study = None
            age = None
            language = None

        elif "prolific" in participantnr:
            participant_type = "prolific"
            participant_id = None
            research_study = None
            age = age_mapper(full_rows, arc_rows)
            language = None

        else:
            continue

        data_dict["type"].append(participant_type)
        data_dict["participant_id"].append(participant_id)
        data_dict["participant_id_arc"].append(arc_user_id)
        data_dict["research_study"].append(research_study)
        data_dict["age"].append(age)
        data_dict["language"].append(language)

    merged_df = pd.DataFrame(data_dict)

    merged_df["participant_id"] = pd.to_numeric(
        merged_df["participant_id"],
        errors="coerce"
    ).astype("Int64")

    merged_df = merged_df.sort_values(
        ["participant_id", "type"],
        na_position="last"
    ).reset_index(drop=True)

    # only keep data from data collection events
    merged_df = merged_df[merged_df["type"] == "event"].drop(columns=["type"])
    # remove age > 99, and age < 3: likely mistakes or not of interest
    merged_df = merged_df[(merged_df["age"] <= 99) & (merged_df["age"] >= 3)]
    return merged_df


if __name__ == "__main__":
    # Import data from server
    merged_df = merge_users(pd.read_csv(ARC_USERS_FILE), pd.read_csv(FULL_USERS_FILE))
    merged_df.to_csv(OUT_FILE, index=False)
    print(f"Merged data saved to '{OUT_FILE}'")
