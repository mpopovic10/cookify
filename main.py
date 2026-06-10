"""
Main entry point for the Cookify recipe recommendation system.
Loads data, runs EDA, trains models and tests recommendations.
"""

from src.data_loader import load_dataset
from src.models.word2vec_model import build_corpus, train_w2v, build_recipe_matrix
from src.models.tfidf_model import train_tfidf
from src.models.recommender import recommend_w2v, recommend_tfidf, recommend_sbert
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
    plot_sbert_embeddings
)

from src.utils import report_missing_ingredients, analyze_missing_ingredients, print_analysis_results
from src.preprocessing import build_ingredient_counter

from src.evaluation import compare_models

#Load data
df = load_dataset()
print(f"Recipes: {len(df)} | Features: {df.shape[1]}")
print(f"Missing values:\n{df.isnull().sum()}")

#EDA
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

#Test queries
test_queries = [
    ["chicken", "lemon", "garlic", "olive oil"],
    ["chocolate", "butter", "sugar", "eggs"],
    ["salmon", "dill", "capers"],
    ["tomato", "basil", "mozzarella"],
    ["flour", "yeast", "salt", "water"],
]

#Experiment 1: Word2Vec
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

#Experiment 2: Word2Vec with lemmatization
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

#Experiment 3: TF-IDF
print("\n--- Experiment 3: TF-IDF ---")
tfidf, matrix_tfidf = train_tfidf(df)
print(f"TF-IDF matrix shape: {matrix_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_tfidf(q, tfidf, matrix_tfidf, df)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))

plot_tfidf_weights(df, matrix_tfidf, tfidf, idx=0)

#Experiment 4: TF-IDF with lemmatization
print("\n--- Experiment 4: TF-IDF with lemmatization ---")
tfidf_lemma, matrix_tfidf_lemma = train_tfidf(df, lemma=True)
print(f"TF-IDF matrix shape: {matrix_tfidf_lemma.shape}")
print(f"Vocabulary size: {len(tfidf_lemma.vocabulary_)}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_tfidf(q, tfidf_lemma, matrix_tfidf_lemma, df, lemma=True)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))


#Experiment 5: Sentence-BERT
print("\n--- Experiment 5: Sentence-BERT ---")

from src.models.sbert_model import build_sbert_text, encode_recipes

df_sbert = build_sbert_text(df)
sbert_model, sbert_embeddings = encode_recipes(df_sbert)
print(f"SBERT embeddings shape: {sbert_embeddings.shape}")

for q in test_queries:
    print(f"\nQuery: {q}")
    res = recommend_sbert(q, sbert_model, sbert_embeddings, df_sbert)
    print(res[["Title", "similarity", "num_matched"]].to_string(index=False))

plot_sbert_embeddings(sbert_embeddings, df)

#User Ingredient Analysis
print("\nUser Ingredient Analysis")

vocab = build_ingredient_counter(df)
vocab_set = set(vocab.keys())

my_ingredients = ["chicken", "lemon", "garlic", "olive oil"]
found = report_missing_ingredients(my_ingredients, vocab_set)

if found:
    results = analyze_missing_ingredients(my_ingredients, df)
    print_analysis_results(results, top_n=5)

#Model Comparison
test_queries_comparison = [
    ["chicken", "lemon", "garlic", "olive oil"],
    ["chocolate", "butter", "sugar", "eggs"],
    ["tomato", "basil", "mozzarella"],
]

for query in test_queries_comparison:
    models_to_compare = {
        "Word2Vec": (recommend_w2v, {'model': w2v, 'df': df_w2v, 'matrix': matrix_w2v}),
        "Word2Vec + Lemma": (recommend_w2v,
                             {'model': w2v_lemma, 'df': df_w2v_lemma, 'matrix': matrix_w2v_lemma, 'lemma': True}),
        "TF-IDF": (recommend_tfidf, {'vectorizer': tfidf, 'matrix': matrix_tfidf, 'df': df}),
        "TF-IDF + Lemma": (recommend_tfidf,
                           {'vectorizer': tfidf_lemma, 'matrix': matrix_tfidf_lemma, 'df': df, 'lemma': True}),
        "SBERT": (recommend_sbert, {'sbert_model': sbert_model, 'embeddings': sbert_embeddings, 'df': df_sbert}),
    }

    results, metrics = compare_models(query, models_to_compare, top_n=5)

#Hybrid Model (TF-IDF + SBERT)

from src.models.hybrid import recommend_pipeline
from src.evaluation import evaluate_hybrid_model

test_queries_hybrid = [
    ["chicken", "lemon", "garlic", "olive oil"],
    ["chocolate", "butter", "sugar", "eggs"],
    ["salmon", "dill", "capers"],
    ["tomato", "basil", "mozzarella"],
    ["flour", "yeast", "salt", "water"],
]

# Test individual queries
for query in test_queries_hybrid[:3]:
    print(f"\nQuery: {query}")
    res = recommend_pipeline(
        query,
        tfidf_vectorizer=tfidf_lemma,
        tfidf_matrix=matrix_tfidf_lemma,
        sbert_model=sbert_model,
        sbert_embeddings=sbert_embeddings,
        df=df_sbert,
        top_n=5,
        candidates=50
    )
    if res is not None:
        print(res[["Title", "tfidf_score", "sbert_score", "num_matched"]].to_string(index=False))

# Evaluate overall performance
hybrid_metrics = evaluate_hybrid_model(
    test_queries_hybrid,
    lambda q, top_n: recommend_pipeline(
        q,
        tfidf_vectorizer=tfidf_lemma,
        tfidf_matrix=matrix_tfidf_lemma,
        sbert_model=sbert_model,
        sbert_embeddings=sbert_embeddings,
        df=df_sbert,
        top_n=top_n,
        candidates=50
    ),
    top_n=5
)