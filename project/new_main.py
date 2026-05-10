import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split

import experiments
import create_datasets_new

# --------------------------
# Чекпоинты
# --------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def process_one_config(num, cfg):
    """
    Обрабатывает один датасет (baseline, smote_variants, imbalanced_learn)
    и сохраняет результат в part_{num:05d}.csv
    """
    part_file = os.path.join(CHECKPOINT_DIR, f"part_{num:05d}.csv")
    # Если чекпоинт уже есть — просто читаем и возвращаем (чтобы общий сбор был корректным)
    if os.path.exists(part_file):
        return pd.read_csv(part_file)

    # 1. Генерация данных
    X, y = create_datasets_new.generate_dataset_from_config(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    # 2. Метаинформация
    dtype = cfg.get("dataset_type", "linear")
    meta = {
        "dataset_id": num,
        "dataset_name": cfg["name"],
        "dataset_type": dtype,
        "ir_level": cfg.get("ir_level", "unknown"),
        "overlap_level": cfg.get("overlap_level", "unknown"),
        "cluster_level": cfg.get("cluster_level") if dtype == "linear" else dtype,
        "flip_y": cfg.get("flip_y", 0.01),
        "n_features": cfg.get("n_features", 20),
        "noise_correlation": cfg.get("noise_correlation", None) if dtype in ("nonlinear","spiral","complex") else None,
        "noise_distribution": cfg.get("noise_distribution", "N/A") if dtype in ("nonlinear","spiral","complex") else "N/A",
    }

    def add_meta(df, group):
        df = df.copy()
        df["group"] = group
        for k, v in meta.items():
            df[k] = v
        return df

    # 3. Три группы экспериментов
    parts = []

    # --- baseline ---
    try:
        base = experiments.run_baseline_experiments(
            X_train, y_train, X_test, y_test,
            model_names=["LogisticRegression", "RandomForest"],
            random_state=42
        )
        parts.append(add_meta(base, "baseline"))
    except Exception as e:
        print(f"Error baseline {cfg['name']}: {e}")

    # --- smote_variants ---
    try:
        sv = experiments.run_experiments(
            X_train, y_train, X_test, y_test,
            sampler_names=["distance_SMOTE", "cluster_SMOTE", "CBSO", "AHC", "DBSMOTE", "MWMOTE"],
            model_names=["LogisticRegression", "RandomForest"],
            verbose=False
        )
        parts.append(add_meta(sv, "smote_variants"))
    except Exception as e:
        print(f"Error sv {cfg['name']}: {e}")

    # --- imbalanced_learn ---
    try:
        imbl = experiments.run_experiments(
            X_train, y_train, X_test, y_test,
            sampler_names=["SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN", "KMeansSMOTE"],
            model_names=["LogisticRegression", "RandomForest"],
            sampler_param_grid={"SMOTE": {"k_neighbors": 3}, "ADASYN": {"n_neighbors": 3}},
            verbose=False
        )
        parts.append(add_meta(imbl, "imbalanced_learn"))
    except Exception as e:
        print(f"Error imbl {cfg['name']}: {e}")

    if not parts:
        # Если всё упало, создаём пустой DataFrame с пометкой
        result = pd.DataFrame({"dataset_id": [num], "error": ["all groups failed"]})
    else:
        result = pd.concat(parts, ignore_index=True)

    # 4. Сохраняем чекпоинт
    result.to_csv(part_file, index=False)
    return result


def collect_all_checkpoints():
    """
    Собирает все part_*.csv в один общий DataFrame и сохраняет в all_results_full.csv.
    Также можно создать отдельные data/model таблицы.
    """
    import glob
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "part_*.csv")))
    if not files:
        return None
    df_list = [pd.read_csv(f) for f in files]
    full_df = pd.concat(df_list, ignore_index=True)
    full_df.to_csv("all_results_full.csv", index=False)
    print(f"Total rows collected: {full_df.shape[0]}")
    return full_df


def main():
    # Загружаем конфиги (можно extended=False для быстрого теста)
    configs = create_datasets_new.get_synthetic_dataset_configs(extended=True)
    print(f"Total configs: {len(configs)}")

    # Определяем уже обработанные по чекпоинтам
    processed = set()
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith("part_") and fname.endswith(".csv"):
            try:
                num = int(fname.split("_")[1].split(".")[0])
                processed.add(num)
            except:
                pass
    print(f"Already processed: {len(processed)}")

    # Формируем список задач (1-based индекс)
    tasks = [(i+1, cfg) for i, cfg in enumerate(configs) if (i+1) not in processed]
    print(f"Remaining tasks: {len(tasks)}")
    if not tasks:
        print("All tasks already done!")
        collect_all_checkpoints()
        return

    # Параллельный запуск
    max_workers = 6   # Укажи нужное число ядер (лучше 6 или меньше, чтобы не перегружать)
    with tqdm(total=len(tasks), desc="Processing datasets") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Словарь future -> номер (для отслеживания)
            future_to_num = {executor.submit(process_one_config, num, cfg): num for num, cfg in tasks}
            for future in as_completed(future_to_num):
                num = future_to_num[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    print(f"Config {num} failed with exception: {exc}")
                pbar.update(1)

    # Финальная сборка
    collect_all_checkpoints()
    print("All done! Final file saved as all_results_full.csv")

if __name__ == "__main__":
    main()