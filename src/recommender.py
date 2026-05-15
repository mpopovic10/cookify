"""
Builds recipe recommendation system using:
cosine similarity between recipe embeddings,
user ingredient preprocessing pipeline,
ranking and returning top-n recipes.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vectorization import recipe_vector
from preprocessing import clean_ingredient_list
from lemmatization import lemmatize_ingredients
from stopwords import remove_stop_words


def build_matrix(df, model):
    vectors = []

    for recipe in df["final_ingredients"]:
        vec = recipe_vector(recipe, model)
        vectors.append(vec)

    return np.vstack(vectors)


def get_recommendations(user_ingredients, df, model, X, top_n=5):
    cleaned = clean_ingredient_list(user_ingredients)
    lemmatized = lemmatize_ingredients([cleaned])[0]
    without_stops = remove_stop_words([lemmatized])[0]

    processed = []
    for word in without_stops:
        if word != "":
            processed.append(word)

    user_vector = recipe_vector(processed, model).reshape(1, -1)

    sims = cosine_similarity(user_vector, X)[0]

    top_indices = np.argsort(sims)[::-1][:top_n]

    results = df.iloc[top_indices][["Title", "final_ingredients"]].copy()
    results["similarity"] = sims[top_indices]

    return results