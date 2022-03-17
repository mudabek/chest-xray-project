import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


# Get the label names from the dataframe
with open('config.yaml') as f:
    config = yaml.safe_load(f)
root_dir = config['path_to_data']

df_valid = pd.read_csv(root_dir + 'valid.csv')
LABEL_NAMES = df_valid.columns[5:]


def f1_as_dict(gt, pred):
    """ Returns F1 scores of all labels as dict for wandb """
    f1_dict = {}
    f1 = f1_score(gt, pred, average=None)
    for i in range(len(f1)):
        f1_dict[f'f1_{LABEL_NAMES[i]}'] = f1[i]

    return f1_dict


def auc_roc_as_dict(y_gt, y_pred):
    """ Returns AUC-ROC scores of all labels as dict for wandb """
    auroc = {}
    gt_np = y_gt.detach().cpu().numpy()
    pred_np = y_pred.detach().cpu().numpy()
    average_auroc = 0

    avging_cnt = 0
    for i in range(gt_np.shape[1]):
        if np.count_nonzero(gt_np[:, i] == 1) > 0:
            cur_roc_auc = roc_auc_score(gt_np[:, i], pred_np[:, i])
            auroc[f'auc_{LABEL_NAMES[i]}'] = cur_roc_auc
            average_auroc =+ cur_roc_auc
            avging_cnt = avging_cnt + 1
    
    auroc['auc_average'] = average_auroc / avging_cnt  

    return auroc
