"""
Loads, samples and cleans the recipe dataset.
"""

import ast
import pandas as pd
from src.config import data_path, sample_size, random_state


def load_raw(path=data_path):
    df = pd.read_csv(path, engine='python', quotechar='"', on_bad_lines='skip')
    df.columns = df.columns.str.strip(';').str.strip()
    return df

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return None


def sample_and_clean(df, n=sample_size):
    """
    Draws a random sample of n rows.
    Drops rows with missing Title or Instructions.
    Removes the unnamed index column.
    Filters out rows whose ingredient lists are empty.
    Parses Cleaned_Ingredients from string into a Python list.
    """
    df = df.sample(n, random_state=random_state).copy()
    df.dropna(subset=["Title", "Instructions", "Cleaned_Ingredients"], inplace=True)
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True)
    df = df[df["Ingredients"] != "[]"]
    df = df[df["Cleaned_Ingredients"] != "['']"]
    df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(safe_literal_eval)
    df = df[df["Cleaned_Ingredients"].notna()]
    return df.reset_index(drop=True)


def add_derived_columns(df):
    """Adds num_ingredients, instr_word_count, and title_word_count columns."""
    df = df.copy()
    df["num_ingredients"] = df["Cleaned_Ingredients"].apply(len)
    df["instr_word_count"] = df["Instructions"].str.split().str.len()
    df["title_word_count"] = df["Title"].str.split().str.len()
    return df


def load_dataset(path=data_path, n=sample_size):
    """Loads, samples and cleans the data.
    Adds derived columns.
    """
    df = load_raw(path)
    df = sample_and_clean(df, n)
    df = add_derived_columns(df)
    return df

load_raw("../data/13k-recipes.csv")