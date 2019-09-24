
import torch
import torch.nn as nn
import numpy as np

from .utils import categorical_to_one_hot

# def segmentation_metrics(pred, target, smooth)


def cal_dice(pred_logit, target, smooth = 1e-8): # target is one hot
    pred = pred_logit.max(dim=1)[1]
    n_classes = pred_logit.shape[1]
    pred_one_hot = categorical_to_one_hot(pred, dim=1, expand_dim=True, n_classes=n_classes)
    
    dice = torch.zeros(n_classes)
    for i_class in range(n_classes):
        dice[i_class] = dice_perclass(pred_one_hot[:,i_class], target[:, i_class], smooth=smooth)
    return dice

def dice_perclass(pred, target, smooth = 1e-8): # both one hot
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum().float() + target.sum() + smooth)

def cal_iou(pred_logit, target, smooth = 1e-8): # target is one hot
    pred = pred_logit.max(dim=1)[1]
    n_classes = pred_logit.shape[1]
    pred_one_hot = categorical_to_one_hot(pred, dim=1, expand_dim=True, n_classes=n_classes)
    
    iou = torch.zeros(n_classes)
    for i_class in range(n_classes):
        iou[i_class] = iou_perclass(pred_one_hot[:,i_class], target[:, i_class], smooth=smooth)
    return iou

def iou_perclass(pred, target, smooth = 1e-8):
    intersection = (pred * target).sum()
    return intersection / (pred.sum().float() + target.sum() - intersection + smooth)



# def dice_coeff_perclass(pred, target, smooth = 1e-8): # both one hot
#     batch_size = pred.shape[0]
#     m1 = pred.view(batch_size, -1)  # Flatten
#     m2 = target.view(batch_size, -1)  # Flatten
#     intersection = (m1 * m2).sum()
    

#     return (2. * intersection) / (m1.sum().float() + m2.sum() + smooth)

# def iou_perclass(pred, target, smooth = 1e-8):
#     batch_size = pred.shape[0]
#     m1 = pred.view(batch_size, -1)  # Flatten
#     m2 = target.view(batch_size, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#     union = ((m1 + m2)>0).sum().float()
#     # if union==0:
#     #     return 0
#     return intersection / (union+smooth)




# def iou(pred_logit, target):
#     n_classes = pred_logit.shape[1]
#     pred = pred_logit.max(dim=1, keepdim=True)[1]
#     pred_one_hot = categorical_to_one_hot(pred, dim=1, n_classes=n_classes).cpu().numpy().astype(bool)
#     y = y.cpu().numpy().astype(bool)

#     # y = y.cpu().numpy().astype(bool)
#     iou = np.zeros((batch_size, n_classes))
#     for i_instance in range(batch_size):
#         for i_class in range(n_classes):
#             if y[i_instance,i_class].sum()==0:
#                 iou[i_instance, i_class] = 0
#             else:    
#                 iou[i_instance, i_class] = ((pred_one_hot[i_instance,i_class] & y[i_instance,i_class]).sum() / 
#                                             (pred_one_hot[i_instance,i_class] | y[i_instance,i_class]).sum())
#     iou = iou.mean(0)

