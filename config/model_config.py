# config/model_config.py
MODEL_CONFIG = {
    'clustering': {
        'n_clusters': 8,
        'algorithm': 'kmeans',
        'features': ['title', 'description_initial', 'domain', 'complexity_level']
    },
    'topic_modeling': {
        'n_topics': 10,
        'method': 'bertopic',
        'language': 'multilingual'  # supports Arabic
    },
    'text_processing': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.95,
        'ngram_range': (1, 2)
    }
}