#загрузка датасетов
from config import DATASETS
from ucimlrepo import fetch_ucirepo 
from tqdm import tqdm


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
 




