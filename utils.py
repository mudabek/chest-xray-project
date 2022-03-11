import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import torch

NUM_FINE_LABELS = 14

def get_class_weights(root_dir):
    df_init = pd.read_csv(root_dir + 'train.csv')
    df = df_init.iloc[:,5:].copy()
    df = df.replace(-1,0)
    class_names = sorted(list(df.columns))
    df = df[class_names]
    pos_weight = torch.Tensor([df[cl].sum()/df.shape[0] for cl in class_names])
    
    return pos_weight

def get_classification_thresholds(out_gt, out_pred):
    all_threshs = []
    for i in range(NUM_FINE_LABELS):
        opt_thres = np.nan
        fpr, tpr, thres = roc_curve(out_gt[:,i], out_pred[:,i])
        pente = tpr - fpr
        opt_thres = thres[np.argmax(pente)]
        all_threshs.append(opt_thres)

    return all_threshs