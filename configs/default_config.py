from datetime import datetime
import os

class ToyShapeConfig():
    # default config
    batch_size = 64
    n_epochs = 50
    drop_rate = 0.0
    seed = None
    num_workers = 0

    # optimizer
    lr = 0.01
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1

    # scheduler
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    gamma=0.1

    save_all = False

    # model 


    # data transform

    flag = '_big0.1'
    save = os.path.join('../tmp', 'toy_shape0.1', datetime.today().strftime("%y%m%d_%H%M%S")+flag)

class ToyVoxelConfig():
    # default config
    batch_size = 16
    n_epochs = 100
    drop_rate = 0.0
    seed = 0
    num_workers = 0

    # optimizer
    lr = 0.001
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    # milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    milestones = [5000]
    gamma=0.1

    conv = 'Conv3D'

    save_all = False

    # model 
    # data transform
    save = os.path.join('../tmp', 'toy_voxel_'+conv, datetime.today().strftime("%y%m%d_%H%M%S")+flag)


class LIDCConfig():
    # default config
    batch_size = 2
    n_epochs = 100
    drop_rate = 0.0
    seed = 0
    num_workers = 0

    # optimizer
    lr = 0.001
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    # milestones = [5000]
    gamma=0.1

    save_all = False

    # model 

    
    # GLOBAL_CONV = 'Conv2_5d'
    # pretrained = True
    # flag = '_pretrained'
    # channels = [64, 128, 256, 512]

    # GLOBAL_CONV = 'Conv3d'
    # pretrained = False
    # flag = '_big'
    # channels = [64, 128, 256, 512]

    # GLOBAL_CONV = 'Conv3d'
    # pretrained = False
    # flag = '_small'
    # channels = [64, 128, 180, 256]

    # GLOBAL_CONV = 'Conv3d'
    # pretrained = False
    # flag = '_bigunet'
    # channels = [64, 128, 256, 384]
    
    save = os.path.join('../tmp', 'LIDC', GLOBAL_CONV, datetime.today().strftime("%y%m%d_%H%M%S")+flag)
