"""
Word2Vec model training and recipe matrix construction.
"""

import numpy as np
from gensim.models import Word2Vec

from src.preprocessing import get_tokens
from src.config import w2v_vector_size, w2v_window, w2v_min_count, w2v_workers, w2v_sg, random_state


def build_corpus(df, lemma=False):
    """
    Builds a list of token lists from the recipe dataset, one per recipe.

    :param df: recipe DataFrame with a Cleaned_Ingredients column
    :param lemma: if True, applies lemmatization during normalisation
    :return: list of token lists
    """
    return [get_tokens(lst, lemma=lemma) for lst in df["Cleaned_Ingredients"]]


def train_w2v(corpus):
    """
    Trains a Word2Vec skip-gram model on the ingredient corpus.

    :param corpus: list of token lists, one per recipe
    :return: trained Word2Vec model
    """
    model = Word2Vec(
        sentences=corpus,
        vector_size=w2v_vector_size,
        window=w2v_window,
        min_count=w2v_min_count,
        workers=w2v_workers,
        sg=w2v_sg,
        seed=random_state,
    )
    return model


def recipe_to_vector(ingredient_list, model, lemma=False):
    """
    Represents a recipe as a single vector by averaging the Word2Vec vectors
    of its ingredient tokens. Returns None if no tokens are found in the vocabulary.

    :param ingredient_list: list of raw ingredient strings
    :param model: trained Word2Vec model
    :param lemma: if True, applies lemmatization during normalisation
    :return: averaged vector as a numpy array or None
    """
    tokens = get_tokens(ingredient_list, lemma=lemma)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def build_recipe_matrix(df, model, lemma=False):
    """
    Computes a vector for each recipe and stacks them into a matrix.

    :param df: recipe DataFrame with a Cleaned_Ingredients column
    :param model: trained Word2Vec model
    :param lemma: if True, applies lemmatization during normalisation
    :return: DataFrame filtered to valid rows and the recipe matrix as a numpy array
    """
    df = df.copy()
    df["w2v_vector"] = df["Cleaned_Ingredients"].apply(
        lambda lst: recipe_to_vector(lst, model, lemma=lemma)
    )
    df_valid = df[df["w2v_vector"].notna()].copy()
    matrix = np.vstack(df_valid["w2v_vector"].values)
    return df_valid, matrix