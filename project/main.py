import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

import metrics
import resampling
import models
import data_loader

from scipy.io import arff

from collections import Counter

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