#загрузка датасетов
from config import DATASETS
from ucimlrepo import fetch_ucirepo 
from tqdm import tqdm
import numpy as np

from collections import Counter
from operator import itemgetter


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
    
def change_overlap():
    pass

def generate_noise():
    pass
    





