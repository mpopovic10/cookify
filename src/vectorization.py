"""
Handles training a Word2Vec model on recipe ingredients,
converting recipes into vector embeddings and
computing average word embeddings for each recipe.
"""

import numpy as np
from gensim.models import Word2Vec

def train_word2vec(data):
    model = Word2Vec(
        sentences=data,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )
    return model


def recipe_vector(recipe, model):
    vectors = []

    for word in recipe:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)