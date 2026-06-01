"""
Exploratory data analysis functions for the recipe dataset.
"""

import pandas as pd
from itertools import combinations

from preprocessing import split_ingredients, normalise, build_ingredient_counter
from src.config import complexity_bins, complexity_labels


def get_top_ingredients(df, n=15, lemma=False):
    """
    Returns the n most common ingredients across all recipes.

    :param df: recipe DataFrame with a Cleaned_Ingredients column
    :param n: number of top ingredients to return
    :param lemma: if True, applies lemmatization during normalisation
    :return: DataFrame with ingredient and count columns
    """
    counter = build_ingredient_counter(df, lemma=lemma)
    return pd.DataFrame(counter.most_common(n), columns=["ingredient", "count"])


def add_complexity_column(df):
    """
    Bins recipes into complexity tiers based on ingredient count.

    :param df: recipe DataFrame with a num_ingredients column
    :return: DataFrame with an added complexity column
    """
    df = df.copy()
    df["complexity"] = pd.cut(
        df["num_ingredients"],
        bins=complexity_bins,
        labels=complexity_labels,
    )
    return df


def complexity_summary(df):
    """
    Returns average, median and count of instruction word count per complexity tier.

    :param df: recipe DataFrame with complexity and instr_word_count columns
    :return: DataFrame with mean, median and count per complexity tier
    """
    return (
        df.groupby("complexity", observed=True)["instr_word_count"]
        .agg(["mean", "median", "count"])
        .round(1)
    )


def correlation_matrix(df):
    """
    Computes the correlation matrix for num_ingredients, instr_word_count and title_word_count.

    :param df: recipe DataFrame with derived columns
    :return: correlation matrix as a DataFrame
    """
    cols = ["num_ingredients", "instr_word_count", "title_word_count"]
    return df[cols].corr()


def cooccurrence_matrix(df, n=10, lemma=False):
    """
    Builds a co-occurrence matrix for the top n ingredients.
    Two ingredients co-occur when they appear together in the same recipe.

    :param df: recipe DataFrame with a Cleaned_Ingredients column
    :param n: number of top ingredients to include
    :param lemma: if True, applies lemmatization during normalisation
    :return: co-occurrence matrix as a DataFrame
    """
    counter = build_ingredient_counter(df, lemma=lemma)
    top_ings = [w for w, _ in counter.most_common(n)]
    cooc = pd.DataFrame(0, index=top_ings, columns=top_ings)

    for lst in df["Cleaned_Ingredients"]:
        present = {
            normalise(p)
            for ing in lst
            for p in split_ingredients(ing)
            if normalise(p) in top_ings
        }
        for a, b in combinations(present, 2):
            cooc.loc[a, b] += 1
            cooc.loc[b, a] += 1

    return cooc