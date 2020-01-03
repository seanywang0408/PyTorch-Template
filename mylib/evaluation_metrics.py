
import torch
import torch.nn as nn
import numpy as np

from .utils import categorical_to_one_hot


def cal_dice_per_case(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).float()
    dice = (2 * intersection / (pred_one_hot.view(batch_size, n_classes, -1).sum(-1).float() + \
                              target.view(batch_size, n_classes, -1).sum(-1).float() + smooth)).mean(0)
    return dice

def cal_batch_dice(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).sum(0).float()
    dice = 2 * intersection / (pred_one_hot.view(batch_size, n_classes, -1).sum(-1).sum(0).float() + \
                             target.view(batch_size, n_classes, -1).sum(-1).sum(0).float() + smooth)
    return dice

def cal_iou_per_case(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).float()
    iou = (intersection / (pred_one_hot.view(batch_size, n_classes, -1).sum(-1).float() + \
                                target.view(batch_size, n_classes, -1).sum(-1).float() - intersection + smooth)).mean(0)
    return iou

def cal_batch_iou(pred_logit, target, smooth = 1e-8): # target is one hot
    pred_classes = pred_logit.max(dim=1)[1]
    batch_size, n_classes = pred_logit.shape[:2]
    pred_one_hot = categorical_to_one_hot(pred_classes, dim=1, expand_dim=True, n_classes=n_classes)
    intersection = (pred_one_hot * target).view(batch_size, n_classes, -1).sum(-1).sum(0).float()
    iou = intersection / ((pred_one_hot.view(batch_size, n_classes, -1).sum(-1).sum(0).float() + \
                          target.view(batch_size, n_classes, -1).sum(-1).sum(0).float() - intersection + smooth))
    return iou
