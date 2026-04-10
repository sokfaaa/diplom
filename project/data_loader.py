#загрузка датасетов
from config import DATASETS
from ucimlrepo import fetch_ucirepo 
from tqdm import tqdm
import numpy as np

from collections import Counter

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

def count_IR(y_train):
    counts = Counter(y_train)
    N_max = max(counts.values())

    basic_IR = N_max/min(counts.values())
    average_IR = N_max / np.mean(counts.values())
    
    per_class_IR = {
        cls: N_max / count for cls, count in counts.items()
    }

    return basic_IR, average_IR, per_class_IR

def change_imbalance_level():
    pass


