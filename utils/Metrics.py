import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
import torch.nn.functional as F
import torch

def dice_coef(y_pred,y_true,num_classes=3,epsilon=1e-7):
    """Altered Sorensen–Dice coefficient with epsilon for smoothing."""
    y_true_flatten = F.one_hot(torch.from_numpy(y_true),num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_flatten = F.one_hot(torch.from_numpy(y_pred),num_classes=num_classes).permute(0, 3, 1, 2)

    if not torch.sum(y_true_flatten) + torch.sum(y_pred_flatten):
        return torch.Tensor(1.0)

    return (2. * torch.sum(y_true_flatten * y_pred_flatten))/(torch.sum(y_true_flatten) + torch.sum(y_pred_flatten) + epsilon)


# def numeric_score(prediction, groundtruth):
#     """Computes scores:
#     FP = False Positives
#     FN = False Negatives
#     TP = True Positives
#     TN = True Negatives
#     return: FP, FN, TP, TN"""

#     FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
#     FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
#     TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
#     TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

#     return FP, FN, TP, TN




def Mereics_score(y_pred, y_true):
    """Getting the accuracy, sensitivity, Specificity, precision F1-score, of the model"""
    mereics_dict=OrderedDict()
    prediction=y_pred.flatten()
    groundtruth=y_true.flatten()
    mereics_dict['accuracy'] =accuracy_score(groundtruth,prediction)
    mereics_dict['precision_score']=precision_score(groundtruth,prediction,average='weighted')
    mereics_dict['recall_score']=recall_score(groundtruth,prediction, average='weighted',zero_division=1)
    mereics_dict['f1_score']=f1_score(groundtruth,prediction, average='weighted')
    return mereics_dict

if __name__ == '__main__':
    a=np.array([0]*10000+[1]*1000+[2]*3000).reshape(1,100,140)
    print(a.shape)
    b=np.zeros_like(a)
    print(Mereics_score(a,b))
    print(dice_coef(a,b))
    