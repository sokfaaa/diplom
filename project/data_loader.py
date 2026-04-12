#загрузка датасетов
from config import DATASETS
from ucimlrepo import fetch_ucirepo 
from tqdm import tqdm
import numpy as np

from collections import Counter
from operator import itemgetter

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

#загрузка данных 

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

#модификация шума, перекрытия, уровня дисбаланса



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



def add_noise(x, sigma=0.1):
    return x + np.random.normal(0, sigma, x.shape)



    





