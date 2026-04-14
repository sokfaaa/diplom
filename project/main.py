import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

import metrics
import resampling
import models
import data

from scipy.io import arff

from collections import Counter

levels = ['easy', 'medium', 'hard']
base_path = "datasets/overlap"
os.makedirs(base_path, exist_ok=True)

for overlap_level in complexity:   
    config = data.build_synthetic_config(n_samples=,
        n_features='medium',
        n_classes='medium',
        imbalance_level="easy",
        overlap_level="easy",
        noise_level="easy",
        cluster_level="easy",
        random_state=42))
        
#x.to_csv('x.csv', index=False)
#y.to_csv('y.csv', index=False)
#x = pd.read_csv("x.csv")
#y = pd.read_csv("y.csv")

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(metrics.n3_error_rate_fast(x_train, y_train))









"""
data, meta = arff.loadarff('dataset_41_glass.arff')
df = pd.DataFrame(data)
df["Type"] = df["Type"].str.decode("utf-8")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df["Type"])
x = df.drop(columns="Type")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(metrics.n3_per_class(x_train, y_train))

metrics.plot_before_resampling('glass', x_train, y_train)



X_new, y_new = data_loader.increase_overlap_adasyn(
    x_train,
    y_train,
    n_neighbors=3,
    sampling_strategy='not majority',
    random_state=42
)
print(metrics.n3_error_rate(X_new, y_new))

metrics.plot_before_resampling('glass_after_overlap_adasyn', X_new, y_new)
"""

"""for k in [2, 5, 10]:
    X_new, y_new = data_loader.increase_overlap_smote(x_train, y_train, k_neighbors=k)
    
    print(f"k_neighbors = {k}")
    print(metrics.n3_error_rate(X_new, y_new))

    metrics.plot_before_resampling(f'glass_after_overlap{k}', X_new, y_new)
"""
"""x_new, y_new = data_loader.increase_multiclass_imblance(x_train, y_train, 0.5)

print(metrics.count_IR(y_new))
print(Counter(y_train), Counter(y_new))"""
"""col = Counter(y_train)
min_key, min_count = min(col.items(), key=itemgetter(1))
print(f"min key:{min_key}, min_count:{min_count}")
print(col)"""
""" 
results = []

for sampler in resampling.samplers:
    for model in models.list_models:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("sampler", sampler),
            ("model", model)
        ])

        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)
        y_proba = pipe.predict_proba(x_test)

        print(metrics.evaluate_model(y_test, y_pred, y_proba))
        metrics.plot_after_resampling(metrics.plot_before_resampling("glass", x_train, y_train),sampler, x_train, y_train)

#result = pd.DataFrame(results)
#print(result.head())
"""