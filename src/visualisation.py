"""
Visualisation functions for the recipe dataset and models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from src.config import figure_dpi, tsne_sample

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = figure_dpi


def plot_top_ingredients(df_top):
    """
    Plots a horizontal bar chart of the most common ingredients.

    :param df_top: DataFrame with ingredient and count columns
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df_top["ingredient"], df_top["count"], color="coral", edgecolor="white")
    ax.invert_yaxis()
    ax.set(title="Top Most Common Ingredients", xlabel="Frequency")
    plt.tight_layout()
    plt.show()


def plot_distributions(df):
    """
    Plots histograms of ingredient count and instruction word count per recipe.

    :param df: recipe DataFrame with num_ingredients and instr_word_count columns
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df["num_ingredients"], bins=30, color="teal", edgecolor="white")
    axes[0].set(title="Ingredients per Recipe", xlabel="Count", ylabel="Recipes")
    axes[1].hist(df["instr_word_count"], bins=40, color="steelblue", edgecolor="white")
    axes[1].set(title="Instruction Length", xlabel="Words", ylabel="Recipes")
    plt.suptitle("Recipe Distributions", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_complexity(df_complexity):
    """
    Plots a bar chart and pie chart of recipe complexity tiers.

    :param df_complexity: Series with complexity tier counts
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(
        df_complexity.index, df_complexity.values,
        color=sns.color_palette("Spectral", 4)
    )
    axes[0].set(title="Recipes by Complexity", ylabel="Recipes")
    axes[1].pie(
        df_complexity.values, labels=df_complexity.index,
        colors=sns.color_palette("Spectral", 4),
        autopct="%1.1f%%", startangle=140
    )
    axes[1].set_title("Complexity Split")
    plt.tight_layout()
    plt.show()


def plot_correlation(corr_matrix):
    """
    Plots a heatmap of the correlation matrix.

    :param corr_matrix: correlation matrix as a DataFrame
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_cooccurrence(cooc_matrix):
    """
    Plots a heatmap of the ingredient co-occurrence matrix.

    :param cooc_matrix: co-occurrence matrix as a DataFrame
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cooc_matrix, mask=np.eye(len(cooc_matrix), dtype=bool),
                annot=True, fmt="d", cmap="YlGnBu", linewidths=0.4, ax=ax)
    ax.set_title("Ingredient Co-occurrence — Top 10")
    plt.tight_layout()
    plt.show()


def plot_tfidf_weights(df, tfidf_matrix, vectorizer, idx=0):
    """
    Plots the top TF-IDF weighted ingredients for a single recipe.

    :param df: recipe DataFrame
    :param tfidf_matrix: fitted TF-IDF matrix
    :param vectorizer: fitted TfidfVectorizer
    :param idx: index of the recipe to visualise
    """
    feature_names = vectorizer.get_feature_names_out()
    sample_vec = tfidf_matrix[idx].toarray()[0]
    top_indices = np.argsort(sample_vec)[::-1][:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        [feature_names[i] for i in top_indices],
        [sample_vec[i] for i in top_indices],
        color="steelblue"
    )
    ax.invert_yaxis()
    ax.set_title(f"Top TF-IDF ingredients: {df['Title'].iloc[idx]}")
    ax.set_xlabel("TF-IDF weight")
    plt.tight_layout()
    plt.show()


def plot_tsne(model, sample=tsne_sample):
    """
    Plots a t-SNE scatterplot of Word2Vec ingredient embeddings.

    :param model: trained Word2Vec model
    :param sample: number of words to sample for the plot
    """
    words = np.random.choice(list(model.wv.key_to_index.keys()), sample)
    word_vectors = np.array([model.wv[w] for w in words])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    twodim = tsne.fit_transform(word_vectors)
    plt.figure(figsize=(14, 14))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors="k", c="r")
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.5, y + 0.5, word, fontsize=8)
    plt.title("t-SNE of Word2Vec ingredient embeddings")
    plt.tight_layout()
    plt.show()


def plot_sbert_embeddings(embeddings, df):
    """
    Visualizes SBERT embeddings using PCA reduction and recipe clusters.

    :param embeddings: recipe embeddings matrix from encode_recipes()
    :param df: recipe DataFrame with Cleaned_Ingredients column
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    def get_recipe_cluster(ingredients_list):
        ingredients_str = " ".join(ingredients_list).lower()
        if 'chocolate' in ingredients_str:
            return 'Chocolate Desserts'
        elif 'chicken' in ingredients_str:
            return 'Chicken Dishes'
        elif 'salmon' in ingredients_str:
            return 'Salmon Meals'
        return 'Other Recipes'

    recipe_clusters = df['Cleaned_Ingredients'].apply(get_recipe_cluster)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=figure_dpi)

    cluster_styles = {
        'Other Recipes': {'color': 'green', 'alpha': 0.15, 'size': 5},
        'Chicken Dishes': {'color': '#E67E22', 'alpha': 0.7, 'size': 20},
        'Chocolate Desserts': {'color': '#4A2711', 'alpha': 0.7, 'size': 20},
        'Salmon Meals': {'color': '#2980B9', 'alpha': 0.8, 'size': 25}
    }

    for cluster_name, style in cluster_styles.items():
        mask = recipe_clusters == cluster_name
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=style['color'],
            label=cluster_name,
            alpha=style['alpha'],
            s=style['size'],
            edgecolors='none'
        )

    ax.set_title('Semantic Space Mapping of Recipe Embeddings (SBERT)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Principal Component 1', fontsize=11)
    ax.set_ylabel('Principal Component 2', fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.show()