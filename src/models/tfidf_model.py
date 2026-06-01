"""
Trains TF-IDF model and builds matrix representation.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import ingredients_to_string


def train_tfidf(df, lemma=False):
    """
    Fits a TF-IDF vectorizer on the recipe dataset and returns the vectorizer
    and the recipe matrix.

    :param df: recipe DataFrame with a Cleaned_Ingredients column
    :param lemma: if True, applies lemmatization during normalisation
    :return: fitted TfidfVectorizer and the TF-IDF matrix
    """
    tfidf_strings = df["Cleaned_Ingredients"].apply(ingredients_to_string, lemma=lemma)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(tfidf_strings)
    return vectorizer, matrix