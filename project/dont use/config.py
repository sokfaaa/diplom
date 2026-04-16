DATASETS = {'glass' : 42, 'dermatology' : 33,'yeast' : 110,
             'letter_recognition' : 59, } #+ 'cover_type' : 31
MODELS = ["random forest", "logistic_regression"]
METHODS = ["smote", "adasyn", "none"]

LEVELS_IMBALANCE = {
    'low' : 0.9,
    'medium' : 0.6,
    'high' : 0.3
}


MINOR_PATTERNS = {
    "easy":   [1, 1, 1, 1, 1, 1, 1],
    "medium": [5, 4, 3, 2, 1, 1, 1],
    "hard":   [9, 7, 5, 3, 2, 1, 1]
}

DEFAULT_MINOR_PATTERN = [7, 5, 3, 2, 1, 1, 1]
 
LEVELS = {
    "n_samples": {
        "easy": 500,
        "medium": 2000,
        "hard": 10000
    },
    "n_classes" : {
        "easy": 3,
        "medium": 5,
        "hard": 8
    }, 
    "n_features": {
        "easy": 10,
        "medium": 20,
        "hard": 50
    },
    
    "overlap": {
        "easy": 2.0,
        "medium": 1.0,
        "hard": 0.5
    },
    "noise": {
        "easy": 0.0,
        "medium": 0.05,
        "hard": 0.1
    },
    "n_clusters": {
        "easy": 1,
        "medium": 2,
        "hard": 3
    },
    "major_weight": {
        "easy": 1/3,
        "medium": 0.6,
        "hard": 0.8
    }
}

