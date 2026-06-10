"""
Model evaluation and comparison utilities.
"""

import pandas as pd


def precision_at_k(df, k=5):
    """
    Precision@K: fraction of top-k recipes that contain at least one matching ingredient.

    :param df: results DataFrame with num_matched column
    :param k: number of top results to evaluate
    :return: precision score (0 to 1)
    """
    top_k = df.head(k)
    relevant = (top_k["num_matched"] > 0).sum()
    return relevant / k


def recall_at_k(df, query_size, k=5):
    """
    Recall@K: fraction of query ingredients recovered across top-k results.

    :param df: results DataFrame with num_matched column
    :param query_size: number of ingredients in the query
    :param k: number of top results to evaluate
    :return: recall score (0 to 1)
    """
    top_k = df.head(k)
    total_matched = top_k["num_matched"].sum()
    max_possible = query_size * k
    return total_matched / max_possible


def compare_models(query_ingredients, models_dict, top_n=5):
    """
    Compares multiple recommendation models on the same query.

    :param query_ingredients: list of ingredient strings
    :param models_dict: dictionary mapping model names to (recommender_func, args)
                        e.g., {"Word2Vec": (recommend_w2v, (w2v, df_w2v, matrix_w2v))}
    :param top_n: number of top recommendations to return and evaluate
    :return: dictionary of results and metrics for each model
    """
    print(f"\nQuery: {query_ingredients}")
    print("=" * 80)

    query_size = len(query_ingredients)
    results = {}
    metrics = {}

    # Run all models
    for model_name, (recommender_func, kwargs) in models_dict.items():
        print(f"\n{model_name}:")
        kwargs['top_n'] = top_n
        res = recommender_func(query_ingredients, **kwargs)
        results[model_name] = res
        print(res[["Title", "similarity", "num_matched"]].to_string(index=False))

    # Calculate metrics
    print("\n" + "-" * 80)
    print("Summary Metrics:")
    print("-" * 80)

    for model_name, res in results.items():
        avg_sim = res["similarity"].mean()
        avg_matched = res["num_matched"].mean()
        avg_coverage = (res["num_matched"] / query_size).mean()

        best_idx = res["num_matched"].idxmax()
        best_recipe_rank = res.index.get_loc(best_idx) + 1

        precision_k = precision_at_k(res, top_n)
        recall_k = recall_at_k(res, query_size, top_n)

        metrics[model_name] = {
            "similarity": avg_sim,
            "matched": avg_matched,
            "coverage": avg_coverage,
            "precision": precision_k,
            "recall": recall_k,
            "best_rank": best_recipe_rank
        }

        print(f"{model_name:25s} | Sim: {avg_sim:.4f} | Matched: {avg_matched:.2f} | "
              f"Coverage: {avg_coverage:.2%} | P@{top_n}: {precision_k:.2f} | R@{top_n}: {recall_k:.2f}")

    # Find best performers
    print("\n" + "-" * 80)
    print("Best Performers:")
    print("-" * 80)

    best_sim = max(metrics, key=lambda m: metrics[m]["similarity"])
    best_matched = max(metrics, key=lambda m: metrics[m]["matched"])
    best_coverage = max(metrics, key=lambda m: metrics[m]["coverage"])
    best_precision = max(metrics, key=lambda m: metrics[m]["precision"])
    best_recall = max(metrics, key=lambda m: metrics[m]["recall"])
    best_rank_model = min(metrics, key=lambda m: metrics[m]["best_rank"])

    print(f"Highest similarity:     {best_sim} ({metrics[best_sim]['similarity']:.4f})")
    print(f"Best ingredient match:  {best_matched} ({metrics[best_matched]['matched']:.2f} avg matched)")
    print(f"Best ingredient coverage: {best_coverage} ({metrics[best_coverage]['coverage']:.2%})")
    print(f"Best Precision@{top_n}:  {best_precision} ({metrics[best_precision]['precision']:.2f})")
    print(f"Best Recall@{top_n}:     {best_recall} ({metrics[best_recall]['recall']:.2f})")
    print(f"Best match rank:        {best_rank_model} (rank {metrics[best_rank_model]['best_rank']})")

    print("=" * 80)

    return results, metrics


def evaluate_hybrid_model(test_queries, recommend_func, top_n=5):
    """
    Evaluates the hybrid recommendation model across multiple test queries.

    :param test_queries: list of test query ingredient lists
    :param recommend_func: the recommendation function to call
    :param top_n: number of recommendations per query
    :return: summary statistics
    """
    precisions = []
    recalls = []
    coverages = []
    ranks = []

    for query in test_queries:
        res = recommend_func(query, top_n=top_n)

        if res is None:
            continue

        query_size = len(query)
        precision = precision_at_k(res, top_n)
        recall = recall_at_k(res, query_size, top_n)
        coverage = (res["num_matched"] / query_size).mean()
        best_idx = res["num_matched"].idxmax()
        best_rank = res.index.get_loc(best_idx) + 1

        precisions.append(precision)
        recalls.append(recall)
        coverages.append(coverage)
        ranks.append(best_rank)

    print("\n" + "=" * 80)
    print("HYBRID MODEL EVALUATION")
    print("=" * 80)
    print(f"Precision@{top_n}:       {np.mean(precisions):.4f}")
    print(f"Recall@{top_n}:          {np.mean(recalls):.4f}")
    print(f"Ingredient Coverage:    {np.mean(coverages):.2%}")
    print(f"Best Match Rank Range:  {min(ranks)}-{max(ranks)}")
    print("=" * 80)

    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "coverage": np.mean(coverages),
        "rank_range": (min(ranks), max(ranks))
    }