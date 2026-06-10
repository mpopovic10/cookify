"""
Hybrid recommendation model: TF-IDF + Lemmatization → SBERT Re-ranking.

Two-stage pipeline:
1. TF-IDF + Lemmatization retrieves top-K candidates (fast, lexical).
2. SBERT re-ranks candidates using semantic similarity (accurate, slower).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalise_lemma, get_tokens


def recommend_pipeline(user_ingredients, tfidf_vectorizer, tfidf_matrix, sbert_model, sbert_embeddings, df, top_n=5,
                       candidates=50):
    """
    Hybrid recommendation using TF-IDF for initial filtering and SBERT for final ranking.

    :param user_ingredients: list of ingredient strings from user
    :param tfidf_vectorizer: fitted TfidfVectorizer (lemmatized)
    :param tfidf_matrix: TF-IDF matrix of recipes (lemmatized)
    :param sbert_model: fitted SentenceTransformer model
    :param sbert_embeddings: SBERT embeddings of all recipes
    :param df: recipe DataFrame with Cleaned_Ingredients and sbert_text columns
    :param top_n: number of final recommendations to return
    :param candidates: number of TF-IDF candidates to re-rank with SBERT
    :return: DataFrame with recommendations, TF-IDF scores, SBERT scores, and matched ingredients
    """

    # Stage 1: TF-IDF + Lemmatization to retrieve candidates
    tokens = []
    for ing in user_ingredients:
        c = normalise_lemma(ing)
        if c and len(c) > 2:
            tokens.append(c)

    if not tokens:
        print("No valid ingredient tokens found. Please try different ingredients.")
        return None

    query_string = ' '.join(tokens)
    query_vec = tfidf_vectorizer.transform([query_string])
    tfidf_sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    candidate_indices = np.argsort(tfidf_sims)[::-1][:candidates]

    # Stage 2: SBERT re-ranking on the candidates
    query_nl = f"Recipe containing {', '.join(user_ingredients)}"
    query_embedding = sbert_model.encode([query_nl])
    candidate_embeddings = sbert_embeddings[candidate_indices]
    sbert_sims = cosine_similarity(query_embedding, candidate_embeddings).flatten()

    # Select top_n from re-ranked candidates
    top_local_indices = np.argsort(sbert_sims)[::-1][:top_n]
    top_global_indices = candidate_indices[top_local_indices]

    # Build results DataFrame
    user_tokens = set(tokens)
    results = df.iloc[top_global_indices][["Title", "Cleaned_Ingredients"]].copy()
    results["tfidf_score"] = tfidf_sims[top_global_indices].round(4)
    results["sbert_score"] = sbert_sims[top_local_indices].round(4)
    results["num_matched"] = results["Cleaned_Ingredients"].apply(
        lambda lst: len(user_tokens.intersection(set(get_tokens(lst, lemma=True))))
    )

    return results.reset_index(drop=True)