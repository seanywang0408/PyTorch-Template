# encoding: utf-8

import _init_paths

import fire
import time
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')


from tqdm import tqdm
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from mylib.utils import MultiAverageMeter, save_model, log_results, to_var, set_seed, \
        to_device, save_prediction, initialize, categorical_to_one_hot, copy_file_backup, \
        plot_multi_shapes, plot_multi_voxels
from mylib.evaluation import cal_iou, cal_dice
from mylib.loss import soft_dice_loss

# from configs.default_config import LIDCConfig as cfg
# from configs.env import LIDCEnv as env

# from models.LIDC.fcn_resnet import FCNResNet18
# from models.LIDC.unet import UNet


def main(save_path=cfg.save, 
         n_epochs=cfg.n_epochs, 
         seed=cfg.seed
         ):
    if seed is not None:
        set_seed(cfg.seed)
    cudnn.benchmark = True

    os.makedirs(save_path)
    copy_file_backup(save_path)

    # Datasets
    train_set = LIDCDataset(crop_size=48, move=5, data_path=env.data, train=True)
    valid_set = None
    test_set = LIDCDataset(crop_size=48, move=5, data_path=env.data, train=False)

    # # Models

    model = FCNResNet18(pretrained=cfg.pretrained, num_classes=2)
    # model = UNet(num_classes=2)

    torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))
    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save_path, n_epochs=n_epochs)

    print('Done!')



def train(model, train_set, test_set, save, valid_set, n_epochs):


    # Data loaders
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                                pin_memory=(torch.cuda.is_available()), num_workers=cfg.num_workers)
    # Model on cuda
    model = to_device(model)

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    # optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=cfg.lr, weight_decay=cfg.wd, momentum=cfg.momentum)
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones,
                                                     gamma=cfg.gamma)

    # Start log
    logs = ['loss', 'iou', 'dice', 'iou0', 'iou1', 'dice0', 'dice1']
    train_logs = ['train_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]

    log_dict = OrderedDict.fromkeys(train_logs+test_logs, 0)
    with open(os.path.join(save, 'logs.csv'), 'w') as f:
        f.write('epoch,')
        for key in log_dict.keys():
            f.write(key+',')
        f.write('\n')
    writer = SummaryWriter(log_dir=os.path.join(save, 'Tensorboard_Results'))

    # Train model
    best_iou = 0

    for epoch in range(n_epochs):
        os.makedirs(os.path.join(cfg.save, 'epoch_{}'.format(epoch)))
        train_meters = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            writer=writer
        )
        test_meters = test_epoch(
            model=model_wrapper,
            loader=test_loader,
            epoch=epoch,
            is_test=True,
            writer = writer
        )
        scheduler.step()

        # Log results
        for i, key in enumerate(train_logs):
            log_dict[key] = train_meters[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_meters[i]

        log_results(save, epoch, log_dict, writer=writer)

        if cfg.save_all:
            torch.save(model.state_dict(), os.path.join(save, 'epoch_{}'.format(epoch), 'model.dat'))

        if log_dict['test_iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
            best_iou = log_dict['test_iou']
            print('New best iou: %.4f' % log_dict['test_iou'])
    writer.close()

    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write(',,,,best iou,%0.5f\n' % (best_iou))
    # Final test of the best model on test set
    print('best iou: ', best_iou)


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1, writer=None):
    meters = MultiAverageMeter()
    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        # Create vaiables
        x = to_var(x)
        y = to_var(y)

        # compute output
        pred_logit = model(x)
        y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)

        loss = soft_dice_loss(pred_logit, y_one_hot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        iou = cal_iou(pred_logit, y_one_hot)
        dice = cal_dice(pred_logit, y_one_hot)

        logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                            [iou[i].item() for i in range(len(iou))]+ \
                            [dice[i].item() for i in range(len(dice))]+ \
                            [time.time() - end]
        meters.update(logs, y.size(0))   

        # measure elapsed time
        end = time.time()


        # print stats
        print_freq = 2 // meters.val[-1] + 1
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
            ])
            print(res)
    pred_one_hot = categorical_to_one_hot(pred_logit.argmax(dim=1), dim=1, expand_dim=True)
    plot_multi_voxels(pred_one_hot[0], y_one_hot[0])
    plt.savefig(os.path.join(cfg.save,'epoch_{}'.format(epoch),'train_{}.pdf'.format(epoch)))
    plt.close()
    return meters.avg[:-1]


def test_epoch(model, loader, epoch, print_freq=1, is_test=True, writer=None):
    meters = MultiAverageMeter()
    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            # Create vaiables
            x = to_var(x)
            y = to_var(y)
            # compute output
            pred_logit = model(x)
            y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)

            loss = soft_dice_loss(pred_logit, y_one_hot)
            # measure accuracy and record loss
            iou = cal_iou(pred_logit, y_one_hot)
            dice = cal_dice(pred_logit, y_one_hot)

            logs = [loss.item(), iou[1:].mean(), dice[1:].mean()]+ \
                                [iou[i].item() for i in range(len(iou))]+ \
                                [dice[i].item() for i in range(len(dice))]+ \
                                [time.time() - end]
            meters.update(logs, y.size(0))   

            # measure elapsed time
            end = time.time()

            # print stats
            print_freq = 2 // meters.val[-1] + 1
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (meters.val[-1], meters.avg[-1]),
                    'Loss %.4f (%.4f)' % (meters.val[0], meters.avg[0]),
                    'IOU %.4f (%.4f)' % (meters.val[1], meters.avg[1]),
                    'DICE %.4f (%.4f)' % (meters.val[2], meters.avg[2]),
                ])
                print(res)
    pred_one_hot = categorical_to_one_hot(pred_logit.argmax(dim=1), dim=1, expand_dim=True)
    plot_multi_voxels(pred_one_hot[0], y_one_hot[0])
    plt.savefig(os.path.join(cfg.save,'epoch_{}'.format(epoch),'test_{}.pdf'.format(epoch)))
    plt.close()
    return meters.avg[:-1]

if __name__ == '__main__':
    fire.Fire(main)

