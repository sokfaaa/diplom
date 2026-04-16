import pandas as pd
import experiments
import create_datasets

from sklearn.model_selection import train_test_split
from tqdm import tqdm

configs = create_datasets.get_synthetic_dataset_configs()

all_results = []
all_data_tables = []
all_model_tables = []

for num, cfg in enumerate(tqdm(configs), start=1):
    X, y = create_datasets.generate_dataset_from_config(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    # --------------------------------
    # 0. baseline (без sampler)
    # --------------------------------
    baseline_df = experiments.run_baseline_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        model_names=["LogisticRegression", "RandomForest"],
        random_state=42
    )

    baseline_df["dataset_id"] = num
    baseline_df["dataset_name"] = cfg["name"]
    baseline_df["group"] = "baseline"
    baseline_df["ir_level"] = cfg["ir_level"]
    baseline_df["overlap_level"] = cfg["overlap_level"]
    baseline_df["cluster_level"] = cfg["cluster_level"]

    data_base, model_base = experiments.split_results_tables(baseline_df)

    data_base["dataset_id"] = num
    data_base["dataset_name"] = cfg["name"]
    data_base["group"] = "baseline"
    data_base["ir_level"] = cfg["ir_level"]
    data_base["overlap_level"] = cfg["overlap_level"]
    data_base["cluster_level"] = cfg["cluster_level"]

    model_base["dataset_id"] = num
    model_base["dataset_name"] = cfg["name"]
    model_base["group"] = "baseline"
    model_base["ir_level"] = cfg["ir_level"]
    model_base["overlap_level"] = cfg["overlap_level"]
    model_base["cluster_level"] = cfg["cluster_level"]

    all_results.append(baseline_df)
    all_data_tables.append(data_base)
    all_model_tables.append(model_base)

    # --------------------------------
    # 1. smote-variants
    # --------------------------------
    results_sv = experiments.run_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        sampler_names=[
            "distance_SMOTE",
            "cluster_SMOTE",
            "CBSO",
            "AHC",
            "DBSMOTE",
            "MWMOTE"
        ],
        model_names=["LogisticRegression", "RandomForest"],
        verbose=False
    )

    results_sv["dataset_id"] = num
    results_sv["dataset_name"] = cfg["name"]
    results_sv["group"] = "smote_variants"
    results_sv["ir_level"] = cfg["ir_level"]
    results_sv["overlap_level"] = cfg["overlap_level"]
    results_sv["cluster_level"] = cfg["cluster_level"]

    data_sv, model_sv = experiments.split_results_tables(results_sv)

    data_sv["dataset_id"] = num
    data_sv["dataset_name"] = cfg["name"]
    data_sv["group"] = "smote_variants"
    data_sv["ir_level"] = cfg["ir_level"]
    data_sv["overlap_level"] = cfg["overlap_level"]
    data_sv["cluster_level"] = cfg["cluster_level"]

    model_sv["dataset_id"] = num
    model_sv["dataset_name"] = cfg["name"]
    model_sv["group"] = "smote_variants"
    model_sv["ir_level"] = cfg["ir_level"]
    model_sv["overlap_level"] = cfg["overlap_level"]
    model_sv["cluster_level"] = cfg["cluster_level"]

    all_results.append(results_sv)
    all_data_tables.append(data_sv)
    all_model_tables.append(model_sv)

    # --------------------------------
    # 2. imbalanced-learn
    # --------------------------------
    results_imbl = experiments.run_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        sampler_names=[
            "SMOTE",
            "BorderlineSMOTE",
            "SVMSMOTE",
            "ADASYN",
            "KMeansSMOTE"
        ],
        model_names=["LogisticRegression", "RandomForest"],
        sampler_param_grid={
            "SMOTE": {"k_neighbors": 3},
            "ADASYN": {"n_neighbors": 3}
        },
        verbose=False
    )

    results_imbl["dataset_id"] = num
    results_imbl["dataset_name"] = cfg["name"]
    results_imbl["group"] = "imbalanced_learn"
    results_imbl["ir_level"] = cfg["ir_level"]
    results_imbl["overlap_level"] = cfg["overlap_level"]
    results_imbl["cluster_level"] = cfg["cluster_level"]

    data_imbl, model_imbl = experiments.split_results_tables(results_imbl)

    data_imbl["dataset_id"] = num
    data_imbl["dataset_name"] = cfg["name"]
    data_imbl["group"] = "imbalanced_learn"
    data_imbl["ir_level"] = cfg["ir_level"]
    data_imbl["overlap_level"] = cfg["overlap_level"]
    data_imbl["cluster_level"] = cfg["cluster_level"]

    model_imbl["dataset_id"] = num
    model_imbl["dataset_name"] = cfg["name"]
    model_imbl["group"] = "imbalanced_learn"
    model_imbl["ir_level"] = cfg["ir_level"]
    model_imbl["overlap_level"] = cfg["overlap_level"]
    model_imbl["cluster_level"] = cfg["cluster_level"]

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