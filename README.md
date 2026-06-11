# Cookify - A Smart Recipe Recommender from Available Ingredients

Cookify is a recipe recommendation system that suggests meals based on ingredients you already have, using text preprocessing and similarity matching.

# Overview

**Cookify** is an ingredient-based recipe recommendation system created to help users decide what to cook using the ingredients they already have. The aim is to save time when choosing meals and reduce food waste. The system utilizes text preprocessing and similarity matching to compare user-input ingredients with a dataset of recipes and recommend the most relevant matches.

- **Dataset:** The model is trained on *Food Ingredients and Recipes Dataset with Images* dataset, which contains over 13,000 different recipes.
-  **Preprocessing:** Recipe ingredients are cleaned and normalized by removing quantities, measurement units and punctuation to ensure consistent ingredient representation.
-  **Similarity Matching:** The system compares user's input ingredients with recipe ingredient sets using similarity measurement to identify and rank the most relevant recipes.
