import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

import metrics
import resampling
import models
import data_loader

from scipy.io import arff

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

print(data_loader.count_IR(y_train))

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