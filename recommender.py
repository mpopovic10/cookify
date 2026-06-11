import pandas as pd
import pickle
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#from app import all_ingredients
from text_processing import (
    normalise_lemma,
    get_tokens_lemma
)

df_5k = pd.read_csv("../../Downloads/UI - Cookify/UI - Cookify/data/recipes.csv")
with open("../../Downloads/UI - Cookify/UI - Cookify/data/recipe_embeddings.pkl", "rb") as f:
    recipe_embeddings = pickle.load(f)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

tfidf_lemma = TfidfVectorizer()
tfidf_lemma_matrix = tfidf_lemma.fit_transform(df_5k["tfidf_lemma_string"])

#vocabulary = set(tfidf_lemma.vocabulary_.keys())  # ✅ dodaj ovdje

def recommend_pipeline(user_ingredients, top_n=5, candidates=50):
    # Stage 1: TF-IDF + Lemmatization
    tokens = []
    for ing in user_ingredients:
        c = normalise_lemma(ing)
        if c and len(c) > 2:
            tokens.append(c)

    if not tokens:
        print("No valid ingredient tokens found. Please try different ingredients.")
        return None


    query_string = ' '.join(tokens)
    query_vec = tfidf_lemma.transform([query_string])

    tfidf_sims = cosine_similarity(query_vec, tfidf_lemma_matrix)[0]
    candidate_idx = np.argsort(tfidf_sims)[::-1][:candidates]

    # Stage 2: SBERT re-ranking on the candidates
    query_nl = f"Recipe containing {', '.join(user_ingredients)}"
    query_embedding = sbert_model.encode([query_nl])

    candidate_embeddings = recipe_embeddings[candidate_idx]
    sbert_sims = cosine_similarity(query_embedding, candidate_embeddings).flatten()

    # Pick top_n from the re-ranked candidates
    top_local_idx = np.argsort(sbert_sims)[::-1][:top_n]
    top_global_idx = candidate_idx[top_local_idx]

    # Build results using b) structure
    user_tokens = set(tokens)
    results = []

    for rank, (local_i, global_i) in enumerate(zip(top_local_idx, top_global_idx), 1):
        row = df_5k.iloc[global_i]
        recipe_tokens   = set(get_tokens_lemma(row["Cleaned_Ingredients"]))
        matched_tokens  = user_tokens.intersection(recipe_tokens)

        precision     = len(matched_tokens) / len(recipe_tokens) if recipe_tokens else 0.0
        recall        = len(matched_tokens) / len(user_tokens)   if user_tokens   else 0.0
        query_overlap = len(matched_tokens) / len(user_tokens)   if user_tokens   else 0.0

        results.append({
            "rank":           rank,
            "title":          row["Title"],
            "image_name":     row["Image_Name"],
            "tfidf_score":    round(float(tfidf_sims[global_i]), 4),
            "sbert_score":    round(float(sbert_sims[local_i]),  4),
            "instructions":   row["Instructions"],
            "ingredients":    ast.literal_eval(row["Cleaned_Ingredients"]),
            "matched_tokens": list(matched_tokens),
            "precision":      round(precision,     4),
            "recall":         round(recall,         4),
            "query_overlap":  round(query_overlap,  4),
        })

    return {
        "query":        user_ingredients,
        "query_tokens": list(user_tokens),
        "results":      results,
    }