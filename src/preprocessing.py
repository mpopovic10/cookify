"""
Handles raw dataset cleaning, including:
loading the dataset, cleaning ingredient text, removing noise words and formatting ingredients.
"""

import pandas as pd
import re
import ast

noise_words = set([
    "finely", "chopped", "freshly", "ground", "divided",
    "plus", "more", "about", "large", "small", "medium",
    "optional", "to", "taste", "such", "as", "depending",
    "on", "like", "cut", "into", "pieces", "piece",
    "thinly", "sliced", "removed", "total"
])

def clean_ingredient(text):
    text = text.lower()

    text = re.sub(r'\d+\/\d+|\d+', '', text)
    text = re.sub(r'[¼½¾⅓⅔⅛]', '', text)

    units = ['cup', 'cups', 'tbsp', 'tsp', 'tablespoon', 'teaspoon',
             'pound', 'lb', 'oz', 'ounce', 'gram', 'kg', 'ml']

    for u in units:
        text = re.sub(r'\b' + u + r's?\b', '', text)

    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = text.replace(" and ", " ")

    phrases_to_remove = [
    "inch in diameter",
    "for serving",
    "to serve"
    ]

    for phrase in phrases_to_remove:
      text = text.replace(phrase, "")

    words = text.split()
    cleaned_words = []

    for w in words:
        if w not in noise_words:
            cleaned_words.append(w)

    return " ".join(cleaned_words).strip()

def clean_ingredient_list(ingredient_list):
  cleaned = []

  for i in ingredient_list:
    cleaned_item = clean_ingredient(i)
    cleaned.append(cleaned_item)

  return cleaned

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df['Title'] = df['Title'].fillna('Untitled')
    df['Instructions'] = df['Instructions'].fillna('')

    df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(ast.literal_eval)
    df["ingredients_clean"] = df["Cleaned_Ingredients"].apply(clean_ingredient_list)

    return df