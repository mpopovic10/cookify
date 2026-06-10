"""
Central configuration file that contains all hardcoded constants used across the project.
Edit this file to change data paths, model hyperparameters or visualisation settings.
"""

#data
data_path = "data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
sample_size = 5000
random_state = 42

#preprocessing
min_token_len = 2
units = (
    "cup", "cups", "tbsp", "tsp", "oz", "lb", "lbs", "g", "kg", "ml",
    "clove", "cloves", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
    "ounce", "ounces", "pound", "pounds",
)

#Word2Vec
w2v_vector_size = 100
w2v_window = 5
w2v_min_count = 1
w2v_workers = 4
w2v_sg = 1  # 1 = skip-gram

#SBERT
sbert_model_name = 'all-MiniLM-L6-v2'

#recommender
top_n_default = 5

#visualisation
figure_dpi = 110
tsne_sample = 100
complexity_bins = [0, 5, 10, 15, 100]
complexity_labels = ["Simple", "Moderate", "Complex", "Elaborate"]