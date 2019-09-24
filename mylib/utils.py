import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import pandas as pd
import os
import time
import random
import torch.nn.functional as F
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')

USE_GPU = True
# USE_GPU = False
import collections.abc
container_abcs = collections.abc
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

if USE_GPU and torch.cuda.is_available():
    def to_var(x, requires_grad=False, gpu=None):
        x = x.cuda(gpu)
        return x.requires_grad_(requires_grad)
else:
    def to_var(x, requires_grad=False, gpu=None):
        return x.requires_grad_(requires_grad)

if USE_GPU and torch.cuda.is_available():
    def to_device(x, gpu=None):
        x = x.cuda(gpu)
        return x
else:
    def to_device(x, gpu=None):
        return x

def save_prediction(model, dataloader, save, saved_name, epoch):
    model.eval()
    model = to_device(model)

    results = pd.DataFrame()
    print('generating and saving prediction...')
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x = to_device(x)
            output = model(x).softmax(-1)
            pred = output.max(-1)[1].cpu()

            results_ = pd.DataFrame(torch.stack([pred, y], dim=-1).numpy(), columns=['pred', 'gt'])
            results = pd.concat([results, results_], axis=0)
    results.to_csv(os.path.join(save, saved_name+'_predictions_{}.csv'.format(epoch)))


    classes = results['gt'].unique().tolist()
    classes.sort()
    data = np.zeros((len(classes),len(classes)), dtype=int)

    for row, gt_class in enumerate(classes):
        for column, pred_class in enumerate(classes):
            data[row, column] = ((results['gt'] == gt_class) & (results['pred'] == pred_class)).sum()

    confusion_matrix = pd.DataFrame(data, index=classes, columns=classes)
    confusion_matrix.to_csv(os.path.join(save, saved_name+'_confusion_matrix_{}.csv'.format(epoch)), index_label='gt \ pred')
    
    print('prediction saved')

def save_prediction_segs(model, dataloader, save, saved_name, epoch):
    model.eval()
    model = to_device(model)

    results = pd.DataFrame()
    print('generating and saving prediction...')
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x = to_device(x)
            pred_logit = model(x)

            # measure accuracy and record loss
            batch_size = y.size(0)
            n_classes=pred_logit.shape[1]
            pred = pred_logit.max(dim=1, keepdim=True)[1]
            pred_one_hot = categorical_to_one_hot(pred,dim=1,n_classes=n_classes).cpu().numpy().astype(bool)

            results_ = pd.DataFrame(torch.stack([pred, y], dim=-1).numpy(), columns=['pred', 'gt'])
            results = pd.concat([results, results_], axis=0)
    results.to_csv(os.path.join(save, saved_name+'_predictions_{}.csv'.format(epoch)))


    classes = results['gt'].unique().tolist()
    classes.sort()
    data = np.zeros((len(classes),len(classes)), dtype=int)

    for row, gt_class in enumerate(classes):
        for column, pred_class in enumerate(classes):
            data[row, column] = ((results['gt'] == gt_class) & (results['pred'] == pred_class)).sum()

    confusion_matrix = pd.DataFrame(data, index=classes, columns=classes)
    confusion_matrix.to_csv(os.path.join(save, saved_name+'_confusion_matrix_{}.csv'.format(epoch)), index_label='gt \ pred')
    
    print('prediction saved')


def confusion_matrix(save, pred):
    file = pd.read_csv(os.path.join(save,pred))
    


class TemperatureScheduler():

    def __init__(self, dataloader, epoch_milestones, T_milestones):
        super().__init__()
        assert len(epoch_milestones) == len(T_milestones)
        assert sorted(epoch_milestones) == epoch_milestones
        self.dataloader = dataloader
        self.epoch_milestones = [int(i) for i in epoch_milestones]
        self.epoch = 0
        self.T_milestones = T_milestones
        self.pointer = 0

    def step(self):
        self.epoch += 1
        if self.epoch in self.epoch_milestones:
            self.dataloader.dataset.temperature = self.T_milestones[self.pointer]
            self.pointer += 1


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MultiAverageMeter(object):
    def __init__(self):
        self.meters = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0        

    def update(self,val,n=1):
        if len(self.meters) < len(val):
            for i in range(len(val)-len(self.meters)):
                self.meters.append(AverageMeter())
        for i, meter in enumerate(self.meters):
            meter.update(val[i],n)
        self.val = [meter.val for meter in self.meters]
        self.avg = [meter.avg for meter in self.meters]
        self.sum = [meter.sum for meter in self.meters]
        self.count = [meter.count for meter in self.meters]


# single class classfication
# def log_results(save, epoch, train_loss, train_error, valid_loss, valid_error, writer):
#     with open(os.path.join(save, 'logs.csv'), 'a') as f:
#         f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
#             (epoch + 1),
#             train_loss,
#             train_error,
#             valid_loss,
#             valid_error,
#         ))
#     writer.add_scalar('train_loss', train_loss, epoch)
#     writer.add_scalar('valid_loss', valid_loss, epoch)
#     writer.add_scalar('train_error', train_error, epoch)
#     writer.add_scalar('valid_error', valid_error, epoch)

def log_results(save, epoch, log_dict, writer):
    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write('%03d,'%((epoch + 1),))
        for value in log_dict.values():
            f.write('%0.6f,' % (value,))
        f.write('\n')
    for key, value in log_dict.items():
        writer.add_scalar(key, value, epoch)


# def categorical_to_one_hot(x, n_dims=None):log_pred = F.log_softmax(pred_logit, dim=1)
#     '''Sequence and label.
#     b x 1 => b x n_dims
#     b x seq => b x seq x n_dims.'''
#     if n_dims is None:
#         n_dims = int(torch.max(x)) + 1
#     x_one_hot = torch.zeros(x.size(0), x.size(1), n_dims).scatter_(dim=2, index=x.unsqueeze(-1), value=1.)
#     return x_one_hot.squeeze(1)  # to FloatTensor

def one_hot_to_categorical(x, dim):
    return x.argmax(dim=dim)

def categorical_to_one_hot(x, dim=1, expand_dim=False, n_classes=None):
    '''Sequence and label.
    when dim = -1:
    b x 1 => b x n_classes
    when dim = 1:
    b x 1 x h x w => b x n_classes x h x w'''
    assert (x - x.long().to(x.dtype)).max().item() < 1e-6
    x = x.long()
    if n_classes is None:
        n_classes = int(torch.max(x)) + 1
    if expand_dim:
        x = x.unsqueeze(dim)
    else:
        assert x.shape[dim] == 1
    shape = list(x.shape)
    shape[dim] = n_classes
    x_one_hot = torch.zeros(shape).to(x.device).scatter_(dim=dim, index=x, value=1.)
    return x_one_hot.long()  




# def final_test(model, save, test_loader):
#     model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
#     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model).cuda()
#     test_results = test_epoch(
#         model=model,
#         loader=test_loader,
#         is_test=True
#     )
#     _, _, test_error = test_results
#     with open(os.path.join(save, 'results.csv'), 'a') as f:
#         f.write(',,,,,%0.5f\n' % (test_error))
#     print('Final test error: %.4f' % test_error)
#     return model

def plot_multi_voxels(*multi_voxels):
    multi_voxels = [np.array(voxels.cpu()) if isinstance(voxels, torch.Tensor) else np.array(voxels) for voxels in multi_voxels]
    multi_voxels = [np.expand_dims(voxels, 0) if voxels.ndim==3 else voxels for voxels in multi_voxels]

    rows = len(multi_voxels[0])
    columns = len(multi_voxels)
    fig = plt.figure(figsize=[10*columns,8*rows])
    for row in range(rows):
        for column in range(columns):
            if row<len(multi_voxels[column]):
                ax = fig.add_subplot(rows,columns,row*columns+column+1, projection='3d')
                ax.voxels(multi_voxels[column][row], edgecolor='k')

def plot_multi_shapes(*multi_shapes):
    multi_shapes = [np.array(shapes.cpu()) if isinstance(shapes, torch.Tensor) else np.array(shapes) for shapes in multi_shapes]
    multi_shapes = [np.expand_dims(shapes, 0) if shapes.ndim==2 else shapes for shapes in multi_shapes]

    rows = len(multi_shapes[0])
    columns = len(multi_shapes)
    fig = plt.figure(figsize=[10*columns,8*rows])
    for row in range(rows):
        for column in range(columns):
            if row<len(multi_shapes[column]):
                ax = fig.add_subplot(rows,columns,row*columns+column+1)
                ax.imshow(multi_shapes[column][row])

def save_model(model, save, valid_error, best_error, save_all):
    # if save_all:
    #     if valid_error < best_error:
    #         best_error = valid_error
    #         print('New best error: %.4f' % best_error)
    #     torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
    # else:
    #     if valid_error < best_error:
    #         best_error = valid_error
    #         print('New best error: %.4f' % best_error)
    #         torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
    if save_all or valid_error < best_error:
        torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
    if valid_error < best_error:
        best_error = valid_error
        print('New best error: %.4f' % best_error)

    return best_error


# def generate_pred(model, dataset, batch_size, save, saved_name, chex_test_transform, num_classes, output_per_model, isDistunc):
#     if dataset is None:
#         print('dataset is none!!!')
#         return

#     cifar_test_process = (lambda model, input: torch.cat(model(input, gen_logits=output_per_model)[
#         0], dim=1)) if isDistunc else (lambda model, input: model(input))

#     def chex_test_process(model, input):
#         bs, n_crops, c, h, w = input.size()
#         input_n_crops = input.view(-1, c, h, w)
#         if torch.cuda.is_available():
#             input_n_crops = input_n_crops.cuda()
#         output = cifar_test_process(model, input_n_crops)
#         output_mean = output.view(bs, n_crops, -1).mean(1)
#         return output_mean

#     test_process = chex_test_process if chex_test_transform else cifar_test_process

#     model.eval()
#     sampler = SequentialSampler(dataset)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
#                                          shuffle=False, pin_memory=(torch.cuda.is_available()), num_workers=0)

#     results = pd.DataFrame()
#     with torch.no_grad():
#         for batch_idx, (input, target) in enumerate(tqdm(loader)):
#             if torch.cuda.is_available():
#                 input = input.cuda()

#             # output = test_process(model, input)
#             output, q_mu, q_sigma = model(input, gen_logits=output_per_model)
#             # if isDistunc:
#             #     output, q_mu, q_sigma = model(
#             #         input, gen_logits=output_per_model)
#             #     output = torch.cat(output, dim=1)
#             # else:
#             #     if output_per_model == 1:
#             #         output = process(model, input)
#             #     else:
#             #         assert model.drop_rate != 0.0
#             #         output = []
#             #         for i in range(output_per_model):
#             #             output.append(process(model, input))
#             #         output = torch.cat(output, dim=1)
#             output = torch.cat(output, dim=-1)
#             results_ = pd.DataFrame(output.cpu().numpy(),
#                                     columns=['output_{}class_{}'.format(_, j) for _ in range(output_per_model) for j in range(num_classes)])
#             results_ = pd.DataFrame(torch.cat([output, q_mu, q_sigma], dim=-1).cpu().numpy(),
#                                     columns=['output_{}class_{}'.format(_, j) for _ in range(output_per_model) for j in range(num_classes)] +
#                                     ['q_mu_{}'.format(j) for j in range(q_mu.size(-1))] +
#                                     ['q_sigma_{}'.format(j) for j in range(q_sigma.size(-1))]
#                                     )
#             results = pd.concat([results, results_], axis=0)
#             # results = pd.concat([results, ], axis=0)
#     results.to_csv(os.path.join(save, saved_name))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            m.bias.data.zero_()



def copy_file_backup(save):
    import shutil
    import sys
    backup_dir = os.path.join(save, 'backup_code')
    os.makedirs(backup_dir)
    with open(os.path.join(backup_dir, 'CLI argument.txt'), 'w') as f:
        for i, _ in enumerate(sys.argv):
            f.write(sys.argv[i] + '\n')
    script_file = os.path.realpath(sys.argv[0])
    shutil.copy(script_file, backup_dir)
    # ignore_list = {'.git', 'scripts', 'tmp', '.ipynb_checkpoints'}
    save_list = {'configs', 'models', 'mylib', 'scripts'}
    for dir_path in os.listdir('../'):
        if os.path.isdir(os.path.join('../', dir_path)) and dir_path in save_list:
            shutil.copytree(os.path.join('../', dir_path), os.path.join(backup_dir, dir_path))


# def dealingWithSaveDir(save):

#     def del_path(path):
#         ls = os.listdir(path)
#         for i in ls:
#             c_path = os.path.join(path, i)
#             if os.path.isdir(c_path):
#                 del_path(c_path)
#             else:
#                 os.remove(c_path)
#         os.rmdir(path)
#     if os.path.exists(save):
#         loop = True
#         print(save)
#         while loop:
#             choice = input(
#                 'The \'save\' dir has already existed. Type \'change\' to change the original dir name as \'[save]_old\'. Type \'new\' to enter a new \'save\' dir. Type \'d\' to delete the original dir. [change/new/delete]? \n')
#             if choice == 'change':
#                 os.rename(save, save + '_Old')

#             elif choice == 'new':
#                 save = input('Please enter a new \'save\' dir name: ')

#             elif choice == 'd':
#                 del_path(save)
#                 print('The original save dir has been deleted!')

#             print('save: ', save)
#             if not os.path.exists(save):
#                 loop = False

#     # Make save directory
#     os.makedirs(save)
#     if not os.path.isdir(save):
#         raise Exception('%s is not a dir' % save)

#     return save


# def test_epoch_cifar(model, loader, save, batch_size, latent_dim, gen_logits=10, print_freq=10, is_test=True, writer=None):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     error = AverageMeter()

#     model.eval()
#     end = time.time()

#     if os.path.exists(os.path.join(save, './test_pred.csv')) and os.path.exists(os.path.join(save, './test_target.csv')):
#         pred_df = pd.read_csv(os.path.join(
#             save, './test_pred.csv'), index_col=0)
#         target_df = pd.read_csv(os.path.join(
#             save, './test_target.csv'), index_col=0)
#         distribution_df = pd.read_csv(os.path.join(
#             save, './test_distribution.csv'), index_col=0)
#     else:
#         pred_df = pd.DataFrame(
#             columns=['test_pred{}'.format(i) for i in range(batch_size)])
#         target_df = pd.DataFrame(
#             columns=['test_target{}'.format(i) for i in range(batch_size)])
#         distribution_df = pd.DataFrame(columns=['q_mu{}'.format(
#             i) for i in range(latent_dim)] + ['q_sigma{}'.format(i) for i in range(latent_dim)])

#     with torch.no_grad():
#         for batch_idx, (input, target) in enumerate(loader):
#             if torch.cuda.is_available():
#                 input = input.cuda()
#                 target = target.cuda()

#             output, q_mu, q_sigma = model(input, gen_logits=gen_logits)
#             output = torch.stack(output, dim=0).mean(dim=0)
#             loss = F.cross_entropy(output, target)

#             cur_batch_size = target.size(0)
#             _, pred = output.data.cpu().topk(1, dim=1)
#             error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / cur_batch_size, cur_batch_size)
#             losses.update(loss.item(), cur_batch_size)
#             pred = pred.squeeze().numpy()
#             target = target.squeeze().cpu().numpy()
#             if batch_size == cur_batch_size:
#                 pred_df.loc[len(pred_df)] = pred
#                 target_df.loc[len(target_df)] = target
#                 distribution_df = pd.concat([distribution_df, pd.DataFrame(torch.cat([q_mu, q_sigma], dim=-1).cpu().detach().numpy()[:3],
#                                                                            columns=['q_mu{}'.format(i) for i in range(latent_dim)] + ['q_sigma{}'.format(i) for i in range(latent_dim)])])
#             batch_time.update(time.time() - end)
#             end = time.time()
#             if batch_idx % print_freq == 0:
#                 res = '\t'.join([
#                     'Test' if is_test else 'Valid',
#                     'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
#                     'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
#                     'Loss %.4f (%.4f)' % (losses.val, losses.avg),
#                     'Error %.4f (%.4f)' % (error.val, error.avg),
#                 ])
#                 print(res)
#     pred_df.to_csv(os.path.join(save, './test_pred.csv'))
#     target_df.to_csv(os.path.join(save, './test_target.csv'))
#     distribution_df.to_csv(os.path.join(save, './test_distribution.csv'))

#     return batch_time.avg, losses.avg, error.avg


def divide_dataset(c_ratio, total, ratio_dir='../ratio_target'):
    percent = c_ratio * 100
    path = os.path.join(ratio_dir, str(percent))
    if os.path.exists(path):
        return path
    print('New Ratio Division')
    os.makedirs(path)
    c_num = int(total * c_ratio)
    total_list = np.arange(total)
    c_list = np.sort(np.array(random.sample(list(total_list), c_num)))
    s_list = np.sort(np.array(list(set(total_list) - set(c_list))))
    pd.DataFrame(c_list, columns=['c_dataset']).to_csv(os.path.join(path, 'train_c_dataset.csv'))
    pd.DataFrame(s_list, columns=['s_dataset']).to_csv(os.path.join(path, 'train_s_dataset.csv'))
    return path


