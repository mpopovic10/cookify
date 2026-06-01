"""
Text normalisation and tokenisation utilities for ingredient strings.
Two variants are provided: standard and lemmatized.
"""

import re
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer

from src.config import min_token_len, units

nltk.download("wordnet", quiet=True)
_lemmatizer = WordNetLemmatizer()


def split_ingredients(text):
    """
    Splits compound ingredient strings on commas, "and", "plus", "or".

    :param text: raw ingredient string
    :return: list of split ingredient parts
    """
    return re.split(r',| and | plus | or ', text)


def normalise(x):
    """
    Strips numbers, units and punctuation from an ingredient string.
    Returns None if the result is too short to be meaningful.

    :param x: raw ingredient string
    :return: cleaned string or None
    """
    x = re.sub(r'\d+\.?\d*', '', x.lower())
    x = re.sub(r'\b(' + '|'.join(units) + r')\b', '', x)
    x = re.sub(r'[^a-z\s]', '', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x if len(x) > min_token_len else None


def normalise_lemma(x):
    """
    Modified version of the normalise function with lemmatization.
    After removing numbers, units and punctuation, each token is reduced to its base
    form with WordNetLemmatizer.

    :param x: raw ingredient string
    :return: cleaned and lemmatized string or None
    """
    x = re.sub(r'\d+\.?\d*', '', x.lower())
    x = re.sub(r'\b(' + '|'.join(units) + r')\b', '', x)
    x = re.sub(r'[^a-z\s]', '', x)
    x = re.sub(r'\s+', ' ', x).strip()
    if not x:
        return None
    lemmatized = ' '.join([
        _lemmatizer.lemmatize(word)
        for word in x.split()
        if len(word) > min_token_len
    ])
    return lemmatized if lemmatized else None


def get_tokens(ingredient_list, lemma=False):
    """
    Converts a recipe's ingredient list into a flat list of clean tokens.
    If lemma is True, applies normalise_lemma instead of normalise.

    :param ingredient_list: list of raw ingredient strings from the dataset
    :param lemma: if True, applies lemmatization during normalisation
    :return: list of clean token strings
    """
    norm_fn = normalise_lemma if lemma else normalise
    tokens = []
    for ing in ingredient_list:
        for part in split_ingredients(ing):
            c = norm_fn(part)
            if c and len(c) > min_token_len:
                tokens.append(c)
    return tokens


def ingredients_to_string(ingredient_list, lemma=False):
    """
    Joins all tokens for a recipe into a single space-separated string for TF-IDF.

    :param ingredient_list: list of raw ingredient strings from the dataset
    :param lemma: if True, applies lemmatization during normalisation
    :return: space-separated string of clean tokens
    """
    return ' '.join(get_tokens(ingredient_list, lemma=lemma))


def build_ingredient_counter(df, lemma=False):
    """
    Counts normalised ingredient tokens across all recipes in the dataset.

    :param df: recipe DataFrame with a Cleaned_Ingredients column
    :param lemma: if True, applies lemmatization during normalisation
    :return: dictionary of ingredient tokens and how often they appear across all recipes
    """
    counter = Counter()
    for lst in df["Cleaned_Ingredients"]:
        for token in get_tokens(lst, lemma=lemma):
            counter[token] += 1
    return counter