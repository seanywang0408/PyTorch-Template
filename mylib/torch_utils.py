
## ------------------------- others ----------------------------- ##
import collections.abc
from itertools import repeat

def _ntuple_same(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            assert len(set(x))==1, 'wrong format'
            return tuple(repeat(x[0], n))
    return parse

def _to_ntuple(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            if len(set(x))==1:
                return tuple(repeat(x[0], n))
            else:
                assert len(x)==n, 'wrong format'
                return x
    return parse

## ------------------------- end of others ----------------------------- ##

## ------------------------ torch utils ----------------------------- ##
import os
import torch 
import torch.nn as nn
import random
import numpy as np

USE_GPU = True
# USE_GPU = False
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

def one_hot_to_categorical(x, dim):
    return x.argmax(dim=dim)

def categorical_to_one_hot(x, dim=1, expand_dim=False, n_classes=None):
    '''Sequence and label.
    when dim = -1:
    b x 1 => b x n_classes
    when dim = 1:
    b x 1 x h x w => b x n_classes x h x w'''
    # assert (x - x.long().to(x.dtype)).max().item() < 1e-6
    if type(x)==np.ndarray:
        x = torch.Tensor(x)
    assert torch.allclose(x, x.long().to(x.dtype))
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

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else: 
        print('No seed is specified, therefore seed is collected from system clock.')
def initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            m.bias.data.zero_()

from .sync_batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d
def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model
def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children(): 
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)

def save_model(model, save, valid_error, best_error, save_all):
    if save_all or valid_error < best_error:
        torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
    if valid_error < best_error:
        best_error = valid_error
        print('New best error: %.4f' % best_error)
    return best_error


## ----------------------------- end of torch utils ----------------------------- ##


