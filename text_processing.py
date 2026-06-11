import re
import nltk
import ast
import string
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def split_ing(text):
  return re.split(r',| and | plus | or ', text)

def normalise(x):
  x = re.sub(r'\d+\.?\d*', '', x.lower())  # remove numbers
  x = re.sub(r'\b(cup|cups|tbsp|tsp|oz|lb|lbs|g|kg|ml|clove|cloves|'
             r'tablespoon|tablespoons|teaspoon|teaspoons|ounce|ounces|pound|pounds)\b', '', x)
  x = re.sub(r'[^a-z\s]', '', x)  # keep letters only
  return re.sub(r'\s+', ' ', x).strip() or None

def get_tokens_lemma(ingredient_list):
  """
  Builds the lemmatized corpus using the same structure from the Experiment 1,
  but applies normalise_lemma. Each recipe will be represented as a list of
  lemmatized tokens.
  """
  tokens = []
  for ing in ingredient_list:
    for part in split_ing(ing):
      c = normalise_lemma(part)
      if c and len(c) > 2:
        tokens.append(c)
  return tokens



def normalise_lemma(x):
  """
  Modified version of the normalise function with lemmatization.
  After removing numbers, units and punctuation, each token is reduced to its base
  form with WordNetLemmatizer.
  """
  x = re.sub(r'\d+\.?\d*', '', x.lower())
  x = re.sub(r'\b(cup|cups|tbsp|tsp|oz|lb|lbs|g|kg|ml|clove|cloves|pound|pounds|'
             r'tablespoon|tablespoons|teaspoon|teaspoons|ounce|ounces)\b', '', x)
  x = re.sub(r'[^a-z\s]', '', x)
  x = re.sub(r'\s+', ' ', x).strip()
  if not x:
    return None
  lemmatized = ' '.join([
      lemmatizer.lemmatize(word)
      for word in x.split()
      if len(word) > 2
  ])
  return lemmatized if lemmatized else None


#from text_processing import normalise_lemma, get_tokens_lemma