import time
import os
import sys

class LIDCSegConfig():
    batch_size = 8
    n_epochs = 100
    drop_rate = 0.0
    seed = 0
    num_workers = 8

    optimizer_choice = 'SGD'
    optimizer_lr = 0.1

    scheduler_milestones = [0.75 * n_epochs]
    scheduler_gamma=0.1

    save_all = False
    use_syncbn = True

    save_path = os.path.join(sys.path[0], './tmp', 'LIDC', backbone, conv, time.strftime("%y%m%d_%H%M%S")+flag)

