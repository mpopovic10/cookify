"""
Recipe recommendation functions for Word2Vec and TF-IDF models.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import split_ingredients, normalise, normalise_lemma, get_tokens
from src.config import top_n_default


def _get_query_tokens(user_ingredients, lemma=False):
    """
    Normalises user input ingredients into a flat list of clean tokens.

    :param user_ingredients: list of ingredient strings provided by the user
    :param lemma: if True, applies lemmatization during normalisation
    :return: list of clean token strings
    """
    norm_fn = normalise_lemma if lemma else normalise
    tokens = []
    for ing in user_ingredients:
        for part in split_ingredients(ing):
            c = norm_fn(part)
            if c:
                tokens.append(c)
    return tokens


def _count_matched(ingredient_list, tokens, lemma=False):
    """
    Counts how many of the query tokens appear in a recipe's ingredient list.

    :param ingredient_list: list of raw ingredient strings from the dataset
    :param tokens: list of clean query tokens
    :param lemma: if True, applies lemmatization during normalisation
    :return: number of matched tokens
    """
    norm_fn = normalise_lemma if lemma else normalise
    return sum(
        1 for ing in ingredient_list
        for part in split_ingredients(ing)
        if norm_fn(part) in tokens
    )


def recommend_w2v(user_ingredients, model, df, matrix, top_n=top_n_default, lemma=False):
    """
    Recommends top-N recipes based on cosine similarity between the user's
    ingredient vector and all recipe vectors in the Word2Vec matrix.

    :param user_ingredients: list of ingredient strings provided by the user
    :param model: trained Word2Vec model
    :param df: recipe DataFrame
    :param matrix: recipe matrix as a numpy array
    :param top_n: number of recipes to return
    :param lemma: if True, applies lemmatization during normalisation
    :return: DataFrame with top-N recommended recipes, similarity scores and matched count
    """
    tokens = _get_query_tokens(user_ingredients, lemma=lemma)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    unknown = [t for t in tokens if t not in model.wv]

    if not vecs:
        print("Sorry, we cannot recommend any recipes with your available ingredients.")
        return None
    if unknown:
        print(f"Unknown tokens (ignored): {unknown}.")

    query_vec = np.mean(vecs, axis=0).reshape(1, -1)
    sims = cosine_similarity(query_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]

    results = df.iloc[top_idx][["Title", "Cleaned_Ingredients"]].copy()
    results["similarity"] = sims[top_idx].round(4)
    results["num_matched"] = results["Cleaned_Ingredients"].apply(
        lambda lst: _count_matched(lst, tokens, lemma=lemma)
    )
    return results.reset_index(drop=True)


def recommend_tfidf(user_ingredients, vectorizer, matrix, df, top_n=top_n_default, lemma=False):
    """
    Recommends top-N recipes based on cosine similarity between the user's
    TF-IDF query vector and all recipe vectors in the TF-IDF matrix.

    :param user_ingredients: list of ingredient strings provided by the user
    :param vectorizer: fitted TfidfVectorizer
    :param matrix: TF-IDF recipe matrix
    :param df: recipe DataFrame
    :param top_n: number of recipes to return
    :param lemma: if True, applies lemmatization during normalisation
    :return: DataFrame with top-N recommended recipes, similarity scores and matched count
    """
    tokens = _get_query_tokens(user_ingredients, lemma=lemma)
    query_string = ' '.join(tokens)
    query_vec = vectorizer.transform([query_string])
    sims = cosine_similarity(query_vec, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_n]

    results = df.iloc[top_idx][["Title", "Cleaned_Ingredients"]].copy()
    results["similarity"] = sims[top_idx].round(4)
    results["num_matched"] = results["Cleaned_Ingredients"].apply(
        lambda lst: _count_matched(lst, tokens, lemma=lemma)
    )
    return results.reset_index(drop=True)


def recommend_sbert(query_ingredients, sbert_model, embeddings, df, top_n=5):
    """
    Recommends recipes using Sentence-BERT semantic similarity.

    :param query_ingredients: list of ingredient strings (e.g., ["chicken", "lemon", "garlic"])
    :param sbert_model: fitted SentenceTransformer model
    :param embeddings: recipe embeddings matrix from encode_recipes()
    :param df: recipe DataFrame with sbert_text column
    :param top_n: number of top recommendations to return
    :return: DataFrame with recommendations and similarity scores
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query_string = f"Recipe containing {', '.join(query_ingredients)}"
    query_embedding = sbert_model.encode([query_string])
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_n]
    results = df.iloc[top_indices][["Title", "Cleaned_Ingredients"]].copy()
    results["similarity"] = similarities[top_indices].round(4)
    results["num_matched"] = results["Cleaned_Ingredients"].apply(
        lambda lst: len(set(_get_query_tokens(query_ingredients)).intersection(set(get_tokens(lst))))
    )
    return results.reset_index(drop=True)