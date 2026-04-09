import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score

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

    




#f1 (macro), auc, gmean
#macro recall - насколько хорошо находятся редкие классы
#balanced accuracy  

