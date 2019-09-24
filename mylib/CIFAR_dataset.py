from __future__ import print_function
import random
import pandas as pd
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch
import torch.utils.data as data
from models.densenet import DenseNet
import pickle


class CifarDatasetByList(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_batch_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train_list, transform):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.data = []
        self.labels = []
        self.train_list = train_list
        downloaded_list = self.train_batch_list
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))[train_list]  # convert to HWC
        self.labels = np.array(self.labels)[train_list]

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class CIFAR10_Confidence_Net(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, num_labels, temperature, isensemble,
                 targets_dir='./targets/', num_classes=10,  # renew_labels, models_dir,
                 train=True, transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.transform = transform
        self.train = train  # training set or test set
        self.isensemble = isensemble
        self.temperature = temperature

        if download:
            self.download()

        if not os.path.exists(targets_dir):
            os.makedirs(targets_dir)

        if self.train:
            downloaded_list = self.train_list
            targets_dir = os.path.join(targets_dir, 'train.csv')
        else:
            downloaded_list = self.test_list
            targets_dir = os.path.join(targets_dir, 'test.csv')

        self.data = []
        self.targets = []
        self.labels = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = list(pd.read_csv(
            targets_dir, index_col=0, dtype=np.float32).values)

    def __getitem__(self, index):
        img = self.data[index]
        all_target_logits = self.targets[index]
        # if self.isensemble:
        #     all_target_logits = torch.from_numpy(all_target_logits) / self.temperature
        #     all_target_P = torch.zeros(all_target_logits.shape)
        #     for i in range(self.num_labels):
        #         all_target_P[i * self.num_classes:(i + 1) * self.num_classes] = all_target_logits[i *
        #                                                                                           self.num_classes:(i + 1) * self.num_classes].softmax(dim=-1)
        #     target = all_target_P
        # else:
        #     select_label = 0
        #     target_logits = torch.from_numpy(all_target_logits[
        #         select_label * self.num_classes:(select_label + 1) * self.num_classes])
        #     target = (target_logits / self.temperature).softmax(dim=-1)
        if self.target_average:
            target_ = 0
            for select_label in range(self.num_labels):
                target_logits = torch.from_numpy(all_target_logits[
                    select_label * self.num_classes:(select_label + 1) * self.num_classes])
                target_ += (target_logits / self.temperature).softmax(dim=-1)
            target = target_ / self.num_labels
        else:
            if self.target_random:
                select_label = random.randint(0, self.num_labels - 1)
            else:
                select_label = 0
            target_logits = torch.from_numpy(all_target_logits[
                select_label * self.num_classes:(select_label + 1) * self.num_classes])
            target = (target_logits / self.temperature).softmax(dim=-1)
        # print(self.temperature)
        label = self.labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, label

    def __len__(self):
        return len(self.data)


class CIFAR10_Confidence_Net_RandomEnsemble(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, num_labels, temperature, target_average, target_random,
                 targets_dir='./targets/', num_classes=10,  # renew_labels, models_dir,
                 train=True, transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.transform = transform
        self.train = train  # training set or test set
        self.target_random = target_random
        self.temperature = temperature
        self.target_average = target_average

        if download:
            self.download()

        if not os.path.exists(targets_dir):
            os.makedirs(targets_dir)

        if self.train:
            downloaded_list = self.train_list
            targets_dir = os.path.join(targets_dir, 'train.csv')
        else:
            downloaded_list = self.test_list
            targets_dir = os.path.join(targets_dir, 'test.csv')

        self.data = []
        self.targets = []
        self.labels = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.labels += entry['labels']
                else:
                    self.labels += entry['fine_labels']

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = list(pd.read_csv(
            targets_dir, index_col=0, dtype=np.float32).values)

    def __getitem__(self, index):
        img = self.data[index]
        all_target_logits = self.targets[index]
        # all_target_logits = torch.from_numpy(all_target_logits) / self.temperature
        # all_target_P = torch.zeros(all_target_logits.shape)
        # for i in range(self.num_labels):
        #     all_target_P[i * self.num_classes:(i + 1) * self.num_classes] = all_target_logits[i *
        # self.num_classes:(i + 1) * self.num_classes].softmax(dim=-1)

        if self.target_average:
            target_ = 0
            for select_label in range(self.num_labels):
                target_logits = torch.from_numpy(all_target_logits[
                    select_label * self.num_classes:(select_label + 1) * self.num_classes])
                target_ += (target_logits / self.temperature).softmax(dim=-1)
            target = target_ / self.num_labels
        else:
            if self.target_random:
                select_label = random.randint(0, self.num_labels - 1)
            else:
                select_label = 0
            target_logits = torch.from_numpy(all_target_logits[
                select_label * self.num_classes:(select_label + 1) * self.num_classes])
            target = (target_logits / self.temperature).softmax(dim=-1)
        # print(self.temperature)
        label = self.labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, label

    def __len__(self):
        return len(self.data)


class CIFAR10_soft_target(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, num_labels, temperature, target_average,
                 targets_dir='./targets/', num_classes=10,  # renew_labels, models_dir,
                 train=True, transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.transform = transform
        self.train = train  # training set or test set

        self.temperature = temperature
        self.target_average = target_average

        if download:
            self.download()

        if not os.path.exists(targets_dir):
            os.makedirs(targets_dir)

        if self.train:
            downloaded_list = self.train_list
            targets_dir = os.path.join(targets_dir, 'train.csv')
        else:
            downloaded_list = self.test_list
            targets_dir = os.path.join(targets_dir, 'test.csv')

        self.data = []
        self.targets = []
        self.labels = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = list(pd.read_csv(
            targets_dir, index_col=0, dtype=np.float32).values)

    def __getitem__(self, index):
        img = self.data[index]
        all_target_logits = self.targets[index]
        if self.target_average:
            target_ = 0
            for select_label in range(self.num_labels):
                target_logits = torch.from_numpy(all_target_logits[
                    select_label * self.num_classes:(select_label + 1) * self.num_classes])
                targets_ += (target_logits / self.temperature).softmax(dim=-1)
            target = target_ / self.num_labels
        else:
            select_label = random.randint(0, self.num_labels - 1)
            target_logits = torch.from_numpy(all_target_logits[
                select_label * self.num_classes:(select_label + 1) * self.num_classes])
            target = (target_logits / self.temperature).softmax(dim=-1)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
