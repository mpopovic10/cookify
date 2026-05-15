"""
Performs lemmatization on tokenized recipe ingredients,
reducing words to their root form.
"""

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def smart_lemmatize(word):
    for pos in ['v', 'a', 'n']:
        result = lemmatizer.lemmatize(word, pos=pos)
        if result != word:
            return result
    return word

def lemmatize_ingredients(all_recipes_list):
  new_big_list = []
  for recipe in all_recipes_list:
    clean_recipe = []
    for word in recipe:
      root_word = smart_lemmatize(word)
      clean_recipe.append(root_word)
    new_big_list.append(clean_recipe)

  return new_big_list