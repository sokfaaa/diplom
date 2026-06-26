import pandas as pd
import numpy as np
from scipy.stats import spearmanr  # можно использовать и pearson

def remove_correlated_features(df, threshold=0.95, method='pearson'):
    """
    Удаляет сильно коррелирующие признаки.
    
    Параметры:
        df: pandas DataFrame с признаками (числовыми)
        threshold: порог корреляции (0.95 = 95%)
        method: 'pearson' или 'spearman'
    
    Возвращает:
        DataFrame с удалёнными коррелированными признаками
        список удалённых признаков
    """
    # Копируем, чтобы не испортить оригинал
    df_corr = df.select_dtypes(include=[np.number]).copy()
    
    # Вычисляем корреляционную матрицу
    if method == 'pearson':
        corr_matrix = df_corr.corr().abs()
    else:
        corr_matrix = df_corr.corr(method='spearman').abs()
    
    # Верхний треугольник матрицы (без диагонали)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Находим столбцы, которые нужно удалить
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Удаляем их
    df_reduced = df_corr.drop(columns=to_drop)
    
    print(f"Исходное число признаков: {df_corr.shape[1]}")
    print(f"Удалено признаков: {len(to_drop)}")
    print(f"Осталось: {df_reduced.shape[1]}")
    print("Удалённые признаки:", to_drop)
    
    return df_reduced, to_drop

# Пример использования:
# загрузите ваш CSV
df = pd.read_csv('metafeatures.csv')

# Предположим, что столбец 'dataset_name' и 'group' нужно сохранить отдельно
# а остальные числовые столбцы — это признаки для кластеризации
metadata_cols = ['dataset_name', 'group']
feature_cols = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['float64', 'int64']]
df_features = df[feature_cols]

# Удаляем коррелирующие признаки (порог 0.95, метод Пирсона)
df_features_clean, dropped = remove_correlated_features(df_features, threshold=0.95, method='pearson')

# Если хотите, можно добавить обратно мета-колонки
df_clean = pd.concat([df[metadata_cols], df_features_clean], axis=1)

# Сохраняем очищенный CSV
df_clean.to_csv('metafeatures_clean.csv', index=False)
print("Очищенный файл сохранён как metafeatures_clean.csv")