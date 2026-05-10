import pandas as pd
import experiments
import create_datasets2

from sklearn.model_selection import train_test_split
from tqdm import tqdm

configs = create_datasets2.get_synthetic_dataset_configs()

all_results = []
all_data_tables = []
all_model_tables = []

for num, cfg in enumerate(tqdm(configs), start=1):
    X, y = create_datasets2.generate_dataset_from_config(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    # -------------------------------------------
    # Извлекаем мета-информацию о датасете
    # -------------------------------------------
    dataset_type = cfg.get("dataset_type", "linear")
    ir_level = cfg.get("ir_level", "unknown")
    overlap_level = cfg.get("overlap_level", "unknown")

    # Для линейных датасетов cluster_level берём из конфига (low/medium/high_clusters)
    # Для нелинейных (nonlinear, spiral, complex) используем сам dataset_type как идентификатор структуры
    if dataset_type == "linear":
        cluster_level = cfg.get("cluster_level", "unknown")
    else:
        cluster_level = dataset_type   # "nonlinear", "spiral", "complex"

    meta_fields = {
        "dataset_id": num,
        "dataset_name": cfg["name"],
        "dataset_type": dataset_type,
        "ir_level": ir_level,
        "overlap_level": overlap_level,
        "cluster_level": cluster_level
    }

    # --------------------------------
    # 0. baseline (без sampler)
    # --------------------------------
    baseline_df = experiments.run_baseline_experiments(
        X_train, y_train, X_test, y_test,
        model_names=["LogisticRegression", "RandomForest"],
        random_state=42
    )

    baseline_df["group"] = "baseline"
    for key, val in meta_fields.items():
        baseline_df[key] = val

    data_base, model_base = experiments.split_results_tables(baseline_df)

    data_base["group"] = "baseline"
    for key, val in meta_fields.items():
        data_base[key] = val

    model_base["group"] = "baseline"
    for key, val in meta_fields.items():
        model_base[key] = val

    all_results.append(baseline_df)
    all_data_tables.append(data_base)
    all_model_tables.append(model_base)

    # --------------------------------
    # 1. smote-variants
    # --------------------------------
    results_sv = experiments.run_experiments(
        X_train, y_train, X_test, y_test,
        sampler_names=[
            "distance_SMOTE", "cluster_SMOTE", "CBSO",
            "AHC", "DBSMOTE", "MWMOTE"
        ],
        model_names=["LogisticRegression", "RandomForest"],
        verbose=False
    )

    results_sv["group"] = "smote_variants"
    for key, val in meta_fields.items():
        results_sv[key] = val

    data_sv, model_sv = experiments.split_results_tables(results_sv)

    data_sv["group"] = "smote_variants"
    for key, val in meta_fields.items():
        data_sv[key] = val

    model_sv["group"] = "smote_variants"
    for key, val in meta_fields.items():
        model_sv[key] = val

    all_results.append(results_sv)
    all_data_tables.append(data_sv)
    all_model_tables.append(model_sv)

    # --------------------------------
    # 2. imbalanced-learn
    # --------------------------------
    results_imbl = experiments.run_experiments(
        X_train, y_train, X_test, y_test,
        sampler_names=[
            "SMOTE", "BorderlineSMOTE", "SVMSMOTE",
            "ADASYN", "KMeansSMOTE"
        ],
        model_names=["LogisticRegression", "RandomForest"],
        sampler_param_grid={
            "SMOTE": {"k_neighbors": 3},
            "ADASYN": {"n_neighbors": 3}
        },
        verbose=False
    )

    results_imbl["group"] = "imbalanced_learn"
    for key, val in meta_fields.items():
        results_imbl[key] = val

    data_imbl, model_imbl = experiments.split_results_tables(results_imbl)

    data_imbl["group"] = "imbalanced_learn"
    for key, val in meta_fields.items():
        data_imbl[key] = val

    model_imbl["group"] = "imbalanced_learn"
    for key, val in meta_fields.items():
        model_imbl[key] = val

    all_results.append(results_imbl)
    all_data_tables.append(data_imbl)
    all_model_tables.append(model_imbl)

# ---------------------------------
# Объединяем всё
# ---------------------------------
all_results_df = pd.concat(all_results, ignore_index=True)
all_data_df = pd.concat(all_data_tables, ignore_index=True)
all_model_df = pd.concat(all_model_tables, ignore_index=True)

# ---------------------------------
# Сохраняем
# ---------------------------------
all_results_df.to_csv("all_results_full.csv", index=False)
all_data_df.to_csv("all_data_metrics.csv", index=False)
all_model_df.to_csv("all_model_metrics.csv", index=False)

print("Готово!")
print("all_results_df:", all_results_df.shape)
print("all_data_df:", all_data_df.shape)
print("all_model_df:", all_model_df.shape)