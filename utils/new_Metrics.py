from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score)


def enhanced_mixing_loss(y_pred, y_true, weight, device='cuda', alpha=0.5, n_classes=3):
    smooth = 1.
    y_true_reshape = F.one_hot(
        y_true, n_classes).permute(0, 3, 1, 2).reshape(-1)
    y_pred_reshape = y_pred.reshape(-1)
    # dice loss
    intersection = (y_true_reshape * y_pred_reshape).sum()
    union = (y_true_reshape + y_pred_reshape).sum()
    dice_loss = 1-(2. * intersection + smooth) / (union + smooth)
    cel_loss = nn.CrossEntropyLoss(
        weight=torch.Tensor(weight).to(device))(y_pred, y_true)
    # # focal loss
    # y_pred = torch.clamp(y_pred, epsilon)

    # pt_1 = torch.where(y_true==1, y_pred, torch.ones_like(y_pred)).float()
    # pt_0 = torch.where(y_true==0, y_pred, torch.zeros_like(y_pred)).float()
    # focal_loss = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - \
    #              torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
    return alpha*cel_loss+(1-alpha)*dice_loss


def Mereics_score(y_pred, y_true, n_classes=3):
    import warnings
    warnings.filterwarnings("ignore")
    """
    Getting the score of model, include these element
    accuracy, sensitivity, Specificity, precision F1-score
    AP, dice, iou and mAP
    more Mereics_score will append in the future
    """
    mereics_dict = OrderedDict()
    y_pred = F.one_hot(torch.from_numpy(y_pred.squeeze()),
                       n_classes).permute(2, 0, 1)[1:].numpy()
    y_true = F.one_hot(torch.from_numpy(y_true.squeeze()),
                       n_classes).permute(2, 0, 1)[1:].numpy()
    mAP=0
    for idx in range(y_pred.shape[0]):

        prediction = y_pred[idx]
        groundtruth = y_true[idx]
        union_area = ((prediction*groundtruth) > 0).sum()
        intersection_area = ((prediction+groundtruth) > 0).sum()
        total_area = (prediction > 0).sum()+(groundtruth > 0).sum()
        mereics_dict['accuracy_' +
                     str(idx+1)] = accuracy_score(groundtruth, prediction)
        mereics_dict['precision_score_'+str(idx+1)] = precision_score(
            groundtruth, prediction, average='weighted', zero_division=1)
        mereics_dict['recall_score_'+str(idx+1)] = recall_score(
            groundtruth, prediction, average='weighted', zero_division=1)
        mereics_dict['f1_score_'+str(idx+1)] = f1_score(
            groundtruth, prediction, average='weighted', zero_division=1)
        if intersection_area==0:
            mereics_dict['AP_'+str(idx+1)]=1
        else:
            mereics_dict['AP_'+str(idx+1)] = average_precision_score(
                groundtruth, prediction, average='weighted')
        mAP+=mereics_dict['AP_'+str(idx+1)]
        
        mereics_dict['iou_'+str(idx+1)] = (union_area +
                                           1e-5)/(intersection_area+1e-5)
        mereics_dict['dice_coff_' +
                     str(idx+1)] = (2*union_area+1e-5)/(total_area+1e-5)
    mereics_dict['mAP' ] = mAP/y_pred.shape[0]
    return mereics_dict


if __name__ == '__main__':
    a = np.random.randint(0, 3, (512, 512))
    b = a
    print(Mereics_score(a, b))
