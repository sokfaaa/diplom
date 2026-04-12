import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors



def evaluate_model(y_test, y_pred, y_proba):
    f1 = f1_score(y_test, y_pred, average= "macro")
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    gmean = geometric_mean_score(y_test, y_pred, average='macro')
    #еще добавить сбалансированную точность
    return f1, auc, gmean

def draw_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plot = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_before_resampling(name_dataset, x_train, y_train):
    # уменьшаем размерность
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_train)

    plt.figure(figsize=(6, 5))

    for label in set(y_train):
        plt.scatter(
            X_pca[y_train == label, 0],
            X_pca[y_train == label, 1],
            label=label
        )

    plt.title("До сэмплинга")
    plt.legend()
    plt.savefig(f'{name_dataset}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return pca

def plot_after_resampling(pca, sampler, x_train, y_train):
    X_res, y_res = sampler.fit_resample(x_train, y_train)

    # PCA снова
    X_res_pca = pca.transform(X_res)

    plt.figure(figsize=(6, 5))

    for label in set(y_res):
        plt.scatter(
            X_res_pca[y_res == label, 0],
            X_res_pca[y_res == label, 1],
            label=label
        )

    plt.title("После RandomOverSampler")
    plt.legend()
    plt.savefig(f'{sampler.__class__.__name__}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return Counter(x_train)

def table_of_result():
    pass

def count_IR(y_train):
    counts = Counter(y_train)
    N_max = max(counts.values())

    basic_IR = N_max/min(counts.values())
    #average_IR = N_max / mean(counts.values())
    
    per_class_IR = {
        cls: N_max / count for cls, count in counts.items()
    }

    return basic_IR, per_class_IR


def fisher_feature_overlap(X, y):
    """
    F1: Fisher's discriminant ratio для каждого признака.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
    y : array-like
    
    Returns
    -------
    pd.Series
        Значение Fisher ratio для каждого признака.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        X = np.asarray(X)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    y = np.asarray(y).squeeze()
    classes = np.unique(y)

    overall_mean = np.mean(X, axis=0)

    numerator = np.zeros(X.shape[1], dtype=float)
    denominator = np.zeros(X.shape[1], dtype=float)

    for cls in classes:
        X_c = X[y == cls]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)
        var_c = np.var(X_c, axis=0, ddof=1)

        numerator += n_c * (mean_c - overall_mean) ** 2
        denominator += n_c * var_c

    fisher_ratio = numerator / (denominator + 1e-12)

    return pd.Series(fisher_ratio, index=feature_names).sort_values(ascending=False)


#f1 (macro), auc, gmean
#macro recall - насколько хорошо находятся редкие классы
#balanced accuracy  


def volume_of_overlap_region(X, y):
    """
    F2: overlap по диапазонам значений признака.
    Для многоклассового случая усредняет overlap по всем парам классов.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        X = np.asarray(X)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    y = np.asarray(y).squeeze()
    classes = np.unique(y)

    results = []

    for j in range(X.shape[1]):
        pair_overlaps = []

        for c1, c2 in combinations(classes, 2):
            x1 = X[y == c1, j]
            x2 = X[y == c2, j]

            min1, max1 = np.min(x1), np.max(x1)
            min2, max2 = np.min(x2), np.max(x2)

            intersection = max(0.0, min(max1, max2) - max(min1, min2))
            union = max(max1, max2) - min(min1, min2)

            overlap = intersection / union if union > 0 else 0.0
            pair_overlaps.append(overlap)

        results.append(np.mean(pair_overlaps) if pair_overlaps else 0.0)

    return pd.Series(results, index=feature_names).sort_values(ascending=False)




def n3_error_rate(X, y):
    """
    N3: leave-one-out error rate для 1-NN.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)

    y = np.asarray(y).squeeze()

    n = len(y)
    errors = 0

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)

        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)[0]

        if pred != y_test:
            errors += 1

    return errors / n


def n3_error_rate_fast(X, y):
    """
    Быстрая версия N3 через ближайших соседей.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)

    y = np.asarray(y).squeeze()

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    # indices[:, 0] — сама точка
    # indices[:, 1] — ближайший сосед
    nearest_neighbor_idx = indices[:, 1]
    nearest_neighbor_labels = y[nearest_neighbor_idx]

    error_rate = np.mean(nearest_neighbor_labels != y)
    return error_rate

def n3_per_class(X, y):
    if isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)

    y = np.asarray(y).squeeze()

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    distances, indices = nn.kneighbors(X)
    nearest_neighbor_idx = indices[:, 1]
    nearest_neighbor_labels = y[nearest_neighbor_idx]

    result = {}

    for cls in np.unique(y):
        mask = (y == cls)
        result[cls] = np.mean(nearest_neighbor_labels[mask] != y[mask])

    return pd.Series(result, name="N3_per_class").sort_values(ascending=False)