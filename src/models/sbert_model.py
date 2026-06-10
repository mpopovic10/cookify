"""
Sentence-BERT model for encoding recipes as dense semantic vectors.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import sbert_model_name


def build_sbert_text(df):
    """
    Creates natural language strings from recipe titles and ingredients.
    :param df: recipe DataFrame with Title and Cleaned_Ingredients columns
    :return: DataFrame with added sbert_text column
    """
    df = df.copy()
    df['sbert_text'] = df.apply(
        lambda row: f"{row['Title']}. Ingredients: {', '.join(row['Cleaned_Ingredients'])}",
        axis=1
    )
    return df


def encode_recipes(df):
    """
    Encodes recipes using Sentence-BERT.
    :param df: recipe DataFrame with sbert_text column (from build_sbert_text)
    :return: tuple of (SentenceTransformer model, embeddings matrix)
    """
    model = SentenceTransformer(sbert_model_name)
    embeddings = model.encode(df['sbert_text'].tolist(), show_progress_bar=True)
    return model, embeddings


def encode_query(query_string, model):
    """
    Encodes a single query string using the SBERT model.
    :param query_string: natural language query (e.g., "Recipe with chicken, lemon, garlic")
    :param model: fitted SentenceTransformer model
    :return: query embedding as a numpy array
    """
    return model.encode([query_string])