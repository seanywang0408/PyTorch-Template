# encoding: utf-8

import _init_paths
import fire
import pandas as pd
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mylib.sync_batchnorm import DataParallelWithCallback
from lidc_dataset import LIDCSegDataset
from mylib.utils import AverageMeter, set_seed, to_device, ResultsLogger, \
        categorical_to_one_hot, backup_code, model_to_syncbn
from mylib.evaluation_metrics import cal_batch_iou, cal_batch_dice
from mylib.loss import soft_dice_loss

from lidc_config import LIDCSegConfig as cfg
from lidc_config import LIDCEnv as env

def main(save_path=cfg.save_path):
    # back up your code 
    backup_code(save_path)
    # set seed
    set_seed(cfg.seed)
    # accelaration
    torch.backends.cudnn.benchmark = True

    # Datasets
    train_set = LIDCSegDataset(crop_size=48, move=5, data_path=env.data, train=True)
    test_set = LIDCSegDataset(crop_size=48, move=5, data_path=env.data, train=False)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)

    # Define model
    model_dict = {'resnet18': FCNResNet, 'vgg16': FCNVGG, 'densenet121': FCNDenseNet}
    model = model_dict[cfg.backbone](pretrained=cfg.pretrained, num_classes=2, backbone=cfg.backbone)

    print(model)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))

    # Model on cuda and then wrap model for multi-GPUs, if necessary
    model = to_device(model)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:       
        if cfg.use_syncbn:
            print('Using sync-bn')
            model_wrapper = DataParallelWithCallback(model).cuda()
        else:
            model_wrapper = torch.nn.DataParallel(model).cuda()
    else:
        model_wrapper = model

    # optimizer and scheduler
    optimizer = getattr(torch.optim, cfg.optimizer_choice)(model_wrapper.parameters(), lr=cfg.optimizer_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler_milestones,
                                                     gamma=cfg.scheduler_gamma)

    results_logger = ResultsLogger(save_path, train_log_items=[] , test_log_items=[])

    # train and test the model
    best_dice_global = 0
    global iteration
    iteration = 0
    for epoch in range(n_epochs):
        # os.makedirs(os.path.join(cfg.save, 'epoch_{}'.format(epoch)))
        print('learning rate: ', scheduler.get_lr())

        train_results = train_epoch(model=model_wrapper, loader=train_loader, optimizer=optimizer,
                                    epoch=epoch, results_logger=results_logger)
        test_results = test_epoch(model=model_wrapper, loader=test_loader, epoch=epoch, results_logger=results_logger)
        scheduler.step()

        results_logger.log_epoch(train_results, test_results)

        # save model checkpoint
        if cfg.save_all:
            torch.save(model.state_dict(), os.path.join(save, 'epoch_{}'.format(epoch), 'model.dat'))

        if  > best_dice_global:
            torch.save(model.state_dict(), os.path.join(save, 'best_model.dat'))
            best_dice_global = 
            print('New best global dice: %.4f' % )
        else:
            print('Current best global dice: %.4f' % best_dice_global)
            

    results_logger.close(best_result=best_dice_global)
    print('best global dice: ', best_dice_global)
    print('Done!')


def train_epoch(model, loader, optimizer, epoch, results_logger):
    '''
    One training epoch
    '''
    meters = AverageMeter()
    # Model on train mode
    model.train()
    global iteration
    intersection = 0
    union = 0
    for batch_idx, (x, y) in enumerate(loader):
        x = to_device(x)
        y = to_device(y)
        # forward and backward
        pred_logit = model(x)
        y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)

        loss = soft_dice_loss(pred_logit, y_one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate metrics
        pred_classes = pred_logit.argmax(1)
        intersection += ((pred_classes==1) * (y[:,0]==1)).sum().item()
        union += ((pred_classes==1).sum() + y[:,0].sum()).item()
        batch_size = y.size(0)

        iou = cal_batch_iou(pred_logit, y_one_hot)
        dice = cal_batch_dice(pred_logit, y_one_hot)
        # log
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        with open(os.path.join(cfg.save, 'loss_logs.csv'), 'a') as f:
            f.write('%09d,%0.6f,\n'%((iteration + 1),loss.item(),))
        iteration += 1

        logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                            [iou[i].item() for i in range(len(iou))]+ \
                            [dice[i].item() for i in range(len(dice))]
        meters.update(logs, batch_size)   

        # print stats
        print_freq = 2 // meters.val[-1] + 1
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, cfg.n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
            ])
            print(res)
    dice_global = 2. * intersection / union
    return meters.avg[:-1] + [dice_global]


def test_epoch(model, loader, optimizer, epoch, results_logger):
    '''
    One test epoch
    '''
    meters = AverageMeter()
    model.eval()
    intersection = 0
    union = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = to_device(x)
            y = to_device(y)
            # forward
            pred_logit = model(x)
            # calculate metrics
            y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)
            pred_classes = pred_logit.argmax(1)
            intersection += ((pred_classes==1) * (y[:,0]==1)).sum().item()
            union += ((pred_classes==1).sum() + y[:,0].sum()).item()

            loss = soft_dice_loss(pred_logit, y_one_hot)
            batch_size = y.size(0)

            iou = cal_batch_iou(pred_logit, y_one_hot)
            dice = cal_batch_dice(pred_logit, y_one_hot)

            logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                                [iou[i].item() for i in range(len(iou))]+ \
                                [dice[i].item() for i in range(len(dice))]
            meters.update(logs, batch_size)   

            print_freq = 2 // meters.val[-1] + 1
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                    'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                    'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                    'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
                ])
                print(res)
    dice_global = 2. * intersection / union

    return meters.avg[:-1] + [dice_global]

if __name__ == '__main__':
    fire.Fire(main)
