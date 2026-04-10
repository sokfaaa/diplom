#я хз где конректно, мб тут. надо будет написать функции для измениния датасетов как раз

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours,
    NearMiss, OneSidedSelection
)

def run_sampler_model(sampler, model, x_train, y_train, x_test, y_test, scaler=True):
    steps = []

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("sampler", sampler))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    return pipe, y_pred


samplers = [
    #oversampling
    RandomOverSampler(sampling_strategy='not majority', random_state=42),
    SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=5),
    BorderlineSMOTE(sampling_strategy='not majority', random_state=42, kind='borderline-1'),
    #ADASYN(sampling_strategy='not majority', random_state=42),
    
    #undersampling
    RandomUnderSampler(sampling_strategy='not minority', random_state=42),
    TomekLinks(sampling_strategy='not minority'),
    #EditedNearestNeighbours(sampling_strategy='not minority'),
    NearMiss(sampling_strategy='not minority', version=1),
    OneSidedSelection(sampling_strategy='not minority', random_state=42),
]
