"""
Main entry point for the Cookify recipe recommendation system.
Loads data, runs EDA, trains models and tests recommendations.
"""

from src.data_loader import load_dataset
from src.models.word2vec_model import build_corpus, train_w2v, build_recipe_matrix
from src.models.tfidf_model import train_tfidf
from src.models.recommender import recommend_w2v, recommend_tfidf
from src.analysis import (
    get_top_ingredients,
    add_complexity_column,
    complexity_summary,
    correlation_matrix,
    cooccurrence_matrix,
)
from src.visualisation import (
    plot_top_ingredients,
    plot_distributions,
    plot_complexity,
    plot_correlation,
    plot_cooccurrence,
    plot_tfidf_weights,
    plot_tsne,
)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_dataset()
print(f"Recipes: {len(df)} | Features: {df.shape[1]}")
print(f"Missing values:\n{df.isnull().sum()}")

# ── EDA ───────────────────────────────────────────────────────────────────────
plot_distributions(df)

df_top = get_top_ingredients(df, n=15)
plot_top_ingredients(df_top)

df = add_complexity_column(df)
counts = df["complexity"].value_counts().sort_index()
plot_complexity(counts)
print(complexity_summary(df))

corr = correlation_matrix(df)
plot_correlation(corr)

cooc = cooccurrence_matrix(df, n=10)
plot_cooccurrence(cooc)

# ── Test queries ──────────────────────────────────────────────────────────────
test_queries = [
    ["chicken", "lemon", "garlic", "olive oil"],
    ["chocolate", "butter", "sugar", "eggs"],
    ["salmon", "dill", "capers"],
    ["tomato", "basil", "mozzarella"],
    ["flour", "yeast", "salt", "water"],
]

# ── Experiment 1: Word2Vec ────────────────────────────────────────────────────
print("\n--- Experiment 1: Word2Vec ---")
corpus = build_corpus(df)
w2v = train_w2v(corpus)
df_w2v, matrix_w2v = build_recipe_matrix(df, w2v)
print(f"Vocabulary size: {len(w2v.wv)}")
print(f"Recipe matrix shape: {matrix_w2v.shape}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_w2v(q, w2v, df_w2v, matrix_w2v)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))

plot_tsne(w2v)

# ── Experiment 2: Word2Vec with lemmatization ─────────────────────────────────
print("\n--- Experiment 2: Word2Vec with lemmatization ---")
corpus_lemma = build_corpus(df, lemma=True)
w2v_lemma = train_w2v(corpus_lemma)
df_w2v_lemma, matrix_w2v_lemma = build_recipe_matrix(df, w2v_lemma, lemma=True)
print(f"Vocabulary size: {len(w2v_lemma.wv)}")
print(f"Recipe matrix shape: {matrix_w2v_lemma.shape}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_w2v(q, w2v_lemma, df_w2v_lemma, matrix_w2v_lemma, lemma=True)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))

plot_tsne(w2v_lemma)

# ── Experiment 3: TF-IDF ──────────────────────────────────────────────────────
print("\n--- Experiment 3: TF-IDF ---")
tfidf, matrix_tfidf = train_tfidf(df)
print(f"TF-IDF matrix shape: {matrix_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_tfidf(q, tfidf, matrix_tfidf, df)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))

plot_tfidf_weights(df, matrix_tfidf, tfidf, idx=0)

# ── Experiment 4: TF-IDF with lemmatization ───────────────────────────────────
print("\n--- Experiment 4: TF-IDF with lemmatization ---")
tfidf_lemma, matrix_tfidf_lemma = train_tfidf(df, lemma=True)
print(f"TF-IDF matrix shape: {matrix_tfidf_lemma.shape}")
print(f"Vocabulary size: {len(tfidf_lemma.vocabulary_)}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_tfidf(q, tfidf_lemma, matrix_tfidf_lemma, df, lemma=True)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))
