#загрузка датасетов
from config import DATASETS
from ucimlrepo import fetch_ucirepo 
from tqdm import tqdm
import numpy as np
import pandas as pd

from collections import Counter
from operator import itemgetter

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from sklearn.datasets import make_classification



#из data.py

#загрузка данных из uci (uci в итоге крашнулся пришлось искать данные в других местах)

def load_df(dataset_id):
    df = fetch_ucirepo(id=dataset_id)
    x = df.data.features
    y = df.data.targets
    return x, y

def load_all_df():
    dfs = {}
    for name, dataset_id in tqdm(DATASETS.items()):
        x, y = load_df(dataset_id)
        y = y.squeeze()
        dfs[name] = {"x": x, "y" : y}
    print("Данные загружены", "\n")
    return dfs

#модификация шума и уровня дисбаланса


def increase_imbalance_by_removing_minority(x,y, remove_fraction=0.3, random_state=42):
    """
    Удаляет случайную долю объектов из самого маленького класса.

    Parameters
    ----------
    X : pd.DataFrame
        Признаки.
    y : pd.Series
        Целевая переменная.
    remove_fraction : float
        Доля объектов минорного класса, которую нужно удалить.
        Например, 0.3 = удалить 30%.
    random_state : int
        Для воспроизводимости.

    Returns
    -------
    X_new : pd.DataFrame
    y_new : pd.Series
    """

    if not 0 <= remove_fraction < 1: 
        raise ValueError("remove_fraction должен быть в диапазоне [0,1)")
    
    y = y.squeeze()

    class_counts = Counter(y)
    minority_class, min_count = min(class_counts.items(), key=itemgetter(1))

    minority_indices = np.where(y == minority_class)[0]
    n_minority = len(minority_indices)

    n_remove = int(n_minority * remove_fraction)

    rng = np.random.default_rng(random_state)
    indices_to_remove = rng.choice(minority_indices, size=n_remove, replace=False)

    x_new = x.drop(index=indices_to_remove)
    y_new = np.delete(y, indices_to_remove)

    return x_new, y_new

def increase_multiclass_imblance(x, y, keep_ratio=0.7, random_state=42):
    """
    Для всех классов, кроме самого большого, случайно оставляет keep_ratio объектов.
    """
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ration должен быть в диапозоне (0, 1]")

    y = y.squeeze()
    rng = np.random.default_rng(random_state)

    class_counts  = Counter(y)
    majority_class, max_count = max(class_counts.items(), key=itemgetter(1))

    indices_to_keep = []

    for cls in class_counts.keys():
        cls_indices = np.where(y == cls)[0]

        if cls == majority_class:
            indices_to_keep.extend(cls_indices)
        else:
            n_keep = max(1, int(len(cls_indices) * keep_ratio))
            chosen = rng.choice(cls_indices, size=n_keep, replace=False)
            indices_to_keep.extend(chosen)
    
    indices_to_keep = sorted(indices_to_keep)
    x_new = x.iloc[indices_to_keep].copy()
    y_new = y[indices_to_keep].copy()

    return x_new, y_new

def add_noise(x, sigma=0.1):
    return x + np.random.normal(0, sigma, x.shape)


#модификация overlap (мне вообще не нравится как это работает, передалю)
    

def increase_overlap_smote(
    X,
    y,
    k_neighbors=5,
    sampling_strategy='not majority',
    random_state=42,
    return_array=False
):
    y = np.asarray(y).squeeze()

    class_counts = Counter(y)
    min_class_size = min(class_counts.values())

    max_valid_k = min_class_size - 1
    if max_valid_k < 1:
        raise ValueError("Слишком мало объектов в одном из классов для применения SMOTE.")

    if k_neighbors > max_valid_k:
        print(f"Предупреждение: k_neighbors={k_neighbors} слишком велик, "
              f"заменяю на {max_valid_k}")
        k_neighbors = max_valid_k

    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    X_res, y_res = smote.fit_resample(X, y)

    if isinstance(X, pd.DataFrame) and not return_array:
        X_res = pd.DataFrame(X_res, columns=X.columns)

    return X_res, y_res


def increase_overlap_adasyn(
    X,
    y,
    n_neighbors=5,
    sampling_strategy='not majority',
    random_state=42,
    return_array=False,
    verbose=True
):
    """
    Увеличивает overlap классов с помощью ADASYN.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Матрица признаков.
    y : array-like
        Целевая переменная.
    n_neighbors : int
        Число соседей для ADASYN.
        Чем больше, тем сильнее могут смещаться синтетические точки к границам классов.
    sampling_strategy : str or dict
        Для multiclass лучше использовать 'not majority', 'minority', 'all'
        или словарь {class_label: target_count}.
    random_state : int
        Seed для воспроизводимости.
    return_array : bool
        Если X был DataFrame и return_array=False, вернётся DataFrame.
        Иначе вернётся numpy array.
    verbose : bool
        Печатать ли предупреждения.

    Returns
    -------
    X_res, y_res
    """

    y = np.asarray(y).squeeze()

    class_counts = Counter(y)
    min_class_size = min(class_counts.values())

    # Для поиска соседей нужен хотя бы 1 допустимый сосед
    max_valid_neighbors = min_class_size - 1
    if max_valid_neighbors < 1:
        raise ValueError(
            "Слишком мало объектов в одном из классов для ADASYN."
        )

    if n_neighbors > max_valid_neighbors:
        if verbose:
            print(
                f"Предупреждение: n_neighbors={n_neighbors} слишком велик для "
                f"минимального класса размера {min_class_size}. "
                f"Использую n_neighbors={max_valid_neighbors}."
            )
        n_neighbors = max_valid_neighbors

    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        n_neighbors=n_neighbors
    )

    try:
        X_res, y_res = adasyn.fit_resample(X, y)
    except ValueError as e:
        # Частый кейс ADASYN:
        # "No samples will be generated with the provided ratio settings."
        # Это значит, что для текущей структуры данных ADASYN не может
        # породить новые точки при выбранных настройках.
        raise ValueError(
            f"ADASYN не смог сгенерировать новые объекты: {e}\n"
            "Попробуй:\n"
            "1) уменьшить n_neighbors,\n"
            "2) задать sampling_strategy вручную через dict,\n"
            "3) использовать SMOTE вместо ADASYN."
        ) from e

    if isinstance(X, pd.DataFrame) and not return_array:
        X_res = pd.DataFrame(X_res, columns=X.columns)

    return X_res, y_res




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
    y_train = y_train.squeeze()
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




#f1 (macro), auc, gmean
#macro recall - насколько хорошо находятся редкие классы
#balanced accuracy  







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


    