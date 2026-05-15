"""
Removes common stopwords from recipe ingredient tokens.
"""

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stop_words(recipe_list):
    final_output = []

    for recipe in recipe_list:
        clean_recipe = []
        for word in recipe:

            if word not in stop_words:
                clean_recipe.append(word)
        final_output.append(clean_recipe)

    return final_output