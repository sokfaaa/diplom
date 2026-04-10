from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

list_models = [
    LogisticRegression(max_iter=2000),
    RandomForestClassifier(random_state=42)
]