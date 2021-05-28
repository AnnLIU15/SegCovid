from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def dice_coef(y_pred, y_true, num_classes=3, epsilon=1e-7):
    """Altered Sorensen–Dice coefficient with epsilon for smoothing."""
    if y_pred.shape == y_true.shape:
        y_true_flatten = F.one_hot(torch.from_numpy(
            y_true), num_classes=num_classes).permute(0, 3, 1, 2)
    else:
        y_true_flatten = y_true
    y_pred_flatten = F.one_hot(torch.from_numpy(
        y_pred), num_classes=num_classes).permute(0, 3, 1, 2)

    if not torch.sum(y_true_flatten) + torch.sum(y_pred_flatten):
        return torch.Tensor(1.0)

    return (2. * torch.sum(y_true_flatten * y_pred_flatten))/(torch.sum(y_true_flatten) + torch.sum(y_pred_flatten) + epsilon)


def dice_loss(y_pred, y_true, num_classes=3, epsilon=1e-7):
    return 1-dice_coef(y_pred, y_true, num_classes=num_classes, epsilon=epsilon)


def enhanced_mixing_loss(y_pred,y_true, weight, device='cuda', alpha=0.5, n_classes=3):
    smooth = 1.
    epsilon = 1e-7
    y_true_reshape = F.one_hot(
        y_true, n_classes).permute(0, 3, 1, 2).reshape(-1)
    y_pred_reshape = y_pred.reshape(-1)
    # dice loss
    intersection = (y_true_reshape * y_pred_reshape).sum()
    union = (y_true_reshape + y_pred_reshape).sum()
    dice_loss = 1-(2. * intersection + smooth) / (union + smooth)
    bce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(weight).to(device))(y_pred, y_true)
    # # focal loss
    # y_pred = torch.clamp(y_pred, epsilon)

    # pt_1 = torch.where(y_true==1, y_pred, torch.ones_like(y_pred)).float()
    # pt_0 = torch.where(y_true==0, y_pred, torch.zeros_like(y_pred)).float()
    # focal_loss = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - \
    #              torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
    return alpha*bce_loss+(1-alpha)*dice_loss

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
    mereics_dict = OrderedDict()
    prediction = y_pred.flatten()
    groundtruth = y_true.flatten()
    mereics_dict['accuracy'] = accuracy_score(groundtruth, prediction)
    mereics_dict['precision_score'] = precision_score(
        groundtruth, prediction, average='weighted',zero_division=1)
    mereics_dict['recall_score'] = recall_score(
        groundtruth, prediction, average='weighted', zero_division=1)
    mereics_dict['f1_score'] = f1_score(
        groundtruth, prediction, average='weighted',zero_division=1)
    return mereics_dict


if __name__ == '__main__':
    a = np.array([0]*10000+[1]*1000+[2]*3000).reshape(1, 100, 140)
    print(a.shape)
    b = np.zeros_like(a)
    print(Mereics_score(a, b))
    print(dice_coef(a, b))
