from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

list_models = [
LogisticRegression(
    max_iter=1000,
    class_weight=None,
    random_state=42
),
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
]