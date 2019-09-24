import torch
import torch.nn as nn
import torch.nn.functional as F

LOGEPS = 1.e-6

def soft_dice_loss(logits, targets, smooth=1.0): # targets is one hot
    probs = logits.softmax(dim=1)
    n_classes = logits.shape[1]
    loss = 0
    for i_class in range(n_classes):
        loss += dice_loss_perclass(probs[:,i_class], targets[:,i_class], smooth)
    return loss / n_classes

def dice_loss_perclass(probs, targets, smooth=1.):
    intersection = probs * targets.float()

    return 1 - 2. * (intersection.sum()+smooth) / (probs.sum()+targets.sum()+smooth)






def soft_cross_entropy_loss(pred_logit, target):
    log_pred = F.log_softmax(pred_logit, dim=-1)
    loss = -(log_pred * target).mean()
    return loss


def batch_det(input):
    return torch.cat([torch.unsqueeze(input[i].det(), dim=0) for i in range(input.shape[0])])\



def gaussian_klloss(p_mu, p_sigma, q_mu, q_sigma):
    # return average KLLoss on one sample
    assert p_mu.shape == p_sigma.shape == q_mu.shape == q_sigma.shape
    cov_p = torch.diag_embed(p_sigma)
    q_sigma_inverse = 1 / q_sigma
    q_cov_inverse = torch.diag_embed(q_sigma_inverse)
    batch_dev_KLLoss = (torch.log(torch.prod(q_sigma, dim=-1) / torch.prod(p_sigma, dim=-1)) - p_mu.shape[-1] +
                        torch.sum(torch.diagonal(q_cov_inverse @ cov_p, dim1=-2, dim2=-1), dim=-1) +
                        ((q_mu - p_mu).unsqueeze(dim=-2) @ q_cov_inverse @ (q_mu - p_mu).unsqueeze(dim=-1)).squeeze()) / 2

    return torch.sum(batch_dev_KLLoss) / p_mu.shape[0], batch_dev_KLLoss


def binary_cross_entropy_with_weights(output_P, target, positive_weights):
    bceloss = - ((1 - positive_weights) * target * output_P.log() + positive_weights
                 * (1 - target) * (1 - output_P).log()).mean()
    return bceloss


def confidence_loss(target, pred_logit, confidence, label):
    log_pred = F.log_softmax(pred_logit, dim=-1)
    ce_loss = -(log_pred * target).sum(dim=-1)
    p_target = target[torch.arange(target.size(0)), label].clamp(LOGEPS, 1 - LOGEPS)
    reject_loss = -torch.log(1 - p_target)

    return ce_loss, reject_loss


def confidence_loss_v2(target, pred_logit, confidence, label):
    log_pred = F.log_softmax(pred_logit, dim=-1)
    ce_loss = -(log_pred * target).sum(dim=-1)

    p_target = target[torch.arange(target.size(0)), label]
    reject_loss = -(p_target * confidence.log() + (1 - p_target) * (1 - confidence).log())

    return ce_loss, reject_loss


def confidence_loss_v3(target, pred_logit, confidence, label):
    log_pred = F.log_softmax(pred_logit / (confidence * 10), dim=-1)
    ce_loss = -(log_pred * target).sum(dim=-1)

    reject_loss = torch.zeros(ce_loss.shape)

    return ce_loss, reject_loss


def confidence_loss_v2_noCE(target, confidence, label, alpha, gamma):

    p_target = target[torch.arange(target.size(0)), label]
    conf_loss = -(alpha * p_target * (1 - confidence)**gamma * confidence.log() +
                  (1 - alpha) * (1 - p_target) * confidence**gamma * (1 - confidence).log())

    return conf_loss


def confidence_loss_v2_noCE_CheX(target, confidence, label, alpha, gamma):
    zero_label = (label == 0).to(torch.float32)
    one_label = (label == 1).to(torch.float32)
    # print(one_label)
    confidence_target = zero_label * (1 - target) + one_label * target
    conf_loss = -(alpha * confidence_target * (1 - confidence)**gamma * confidence.log() +
                  (1 - alpha) * (1 - confidence_target) * confidence**gamma * (1 - confidence).log())

    return conf_loss
