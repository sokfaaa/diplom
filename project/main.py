from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import data_loader
import metrics
import tqdm

datasets = data_loader.load_all_df()

for dataset_name, data in datasets.items():
    print(f"Начало обработки датасета {dataset_name}")
    x = data["x"]
    y = data["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    print(metrics.evaluate_model(y_test, y_pred, y_proba))
    metrics.draw_confusion_matrix(dataset_name, y_test, y_pred)

    print(f"Обработана датасет {dataset_name}", "\n")
