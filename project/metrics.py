import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.decomposition import PCA

from collections import Counter

def evaluate_model(y_test, y_pred, y_proba):
    f1 = f1_score(y_test, y_pred, average= "macro")
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    gmean = geometric_mean_score(y_test, y_pred, average='macro')
    return f1, auc, gmean

def draw_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plot = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_before_resampling(name_dataset, x_train, y_train):
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


#f1 (macro), auc, gmean
#macro recall - насколько хорошо находятся редкие классы
#balanced accuracy  

