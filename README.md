# Cookify - A Smart Recipe Recommender from Available Ingredients

Cookify is a recipe recommendation system that suggests meals based on ingredients you already have, using text preprocessing and similarity matching.

# Overview

**Cookify** is an ingredient-based recipe recommendation system created to help users decide what to cook using the ingredients they already have. The aim is to save time when choosing meals and reduce food waste. The system utilizes text preprocessing and similarity matching to compare user-input ingredients with a dataset of recipes and recommend the most relevant matches.

- **Dataset:** The model is trained on *Food Ingredients and Recipes Dataset with Images* dataset, which contains over 13,000 different recipes.
-  **Preprocessing:** Recipe ingredients are cleaned and normalized by removing quantities, measurement units and punctuation to ensure consistent ingredient representation.
-  **Similarity Matching:** The system compares user's input ingredients with recipe ingredient sets using similarity measurement to identify and rank the most relevant recipes.

# Dataset
## Step 1: Data Source
Cookify uses the [*Food Ingredients and Recipes Dataset with Images*](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images) dataset that includes 13,582 recipes, covering a wide variety of different cuisines and ingredient combinations. Each recipe has 5 features:
1) **Title:** Name of the meal
2) **Ingredients:** Contains ingredients in the form they were scrapped from the website
3) **Instructions:** Includes steps to follow when making the recipe
4) **Image Name:** A reference to the meal image in the *Food Images* zip folder
5) **Cleaned Ingredients:** Contains processed and cleaned ingredients

## Exploratory Data Analysis (EDA)

The Exploratory Data Analysis was performed on a random sample of **5,000 recipes** drawn from the full *Food Ingredients and Recipe Dataset with Image Name Mapping* dataset (`random_state=42`).

### 1. Dataset Overview

| Feature | Value |
| :--- | :--- |
| **Rows (recipes)** | 5,000 |
| **Columns** | 5 (`Title`, `Ingredients`, `Instructions`, `Image_Name`, `Cleaned_Ingredients`) |
| **Duplicate Rows** | 0 *(Note: Some recipes share titles but differ in content)* |

### 2. Engineered Features

Three numeric features were created to characterize each recipe:
* `num_ingredients`: Number of items in `Cleaned_Ingredients` (proxy for recipe complexity).
* `instr_word_count`: Word count of `Instructions` (proxy for recipe verbosity).
* `title_word_count`: Word count of `Title`.

| Feature | Mean | Median | Std |
| :--- | :---: | :---: | :---: |
| **num_ingredients** | ~9.5 | 9 | ~4.6 |
| **instr_word_count** | ~155 | 120 | ~130 |
| **title_word_count** | ~4.0 | 4 | ~1.8 |

Both `num_ingredients` and `instr_word_count` are **right-skewed**: most recipes are short and simple, but a long tail of elaborate recipes pulls the mean upward.

<img src="recipe-distributions.png" alt="Feature Length Distributions" width="100%" />
### 3. Recipe Complexity Distribution

Recipes were binned into four complexity tiers based on their ingredient count:

* **Simple** (1–5 ingredients): ~15% share
* **Moderate** (6–10 ingredients): ~50% share
* **Complex** (11–15 ingredients): ~25% share
* **Elaborate** (16+ ingredients): ~10% share

**Moderate recipes dominate the dataset.** Average instruction length grows steadily with complexity (Moderate recipes average ~120 words, while Elaborate recipes average ~250 words), confirming that ingredient count is a highly reliable proxy for overall recipe difficulty.

<img src="recipe-complexity.png" alt="Recipe Complexity Tiers" width="100%" />

### 4. Ingredient Frequency Analysis

Before counting, ingredient strings were pre-processed to reduce text noise by splitting compound entries, normalising to lowercase without units, and filtering short tokens. The final vocabulary contained **~3,700 unique normalized ingredient tokens**.

#### Top 15 Most Frequent Ingredients:
1. **Salt** (2,800+ counts)
2. **Butter** (2,200+ counts)
3. **Sugar** (2,100+ counts)
4. **Olive Oil** (1,900+ counts)
5. **Garlic** (1,800+ counts)
6. **Pepper** (1,700+ counts)
7. **Flour** (1,600+ counts)
8. **Egg** (1,500+ counts)
9. **Onion** (1,400+ counts)
10. **Water** (1,300+ counts)
11. **Milk** (1,100+ counts)
12. **Lemon** (1,000+ counts)
13. **Chicken** (950 counts)
14. **Cream** (900 counts)
15. **Tomato** (850 counts)

<img src="top-most-15-ingredients.png" alt="Top 15 Most Common Ingredients" width="100%" />

### 5. Ingredient Co-occurrence & Correlations

A co-occurrence matrix built for the top 10 ingredients revealed clear culinary patterns:
* **Baking Dominance:** `butter–sugar` and `butter–flour` are the strongest ingredient pairs.
* **Savoriness:** `garlic–olive oil` and `garlic–onion` are the dominant savory co-occurrence pairs.
* **Salt Prevalence:** Salt co-occurs broadly with almost every top ingredient, making it the least discriminative feature for data retrieval.

<img src="ingredients-per-occurance.png" alt="Ingredient Co-occurrence Matrix" width="100%" />

#### Feature Correlations (Pearson r):
* `num_ingredients` ↔ `instr_word_count`: **~0.35** (Moderate positive correlation)
* `num_ingredients` ↔ `title_word_count`: **~0.05** (No correlation)
* `instr_word_count` ↔ `title_word_count`: **~0.02** (No correlation)

<img src="correlation-matrix.png" alt="Correlation Matrix Heatmap" width="100%" />
