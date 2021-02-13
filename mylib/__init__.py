## ----------------------------- training utils ----------------------------- ##
# back up functions
import sys, time, shutil, getpass, socket

def backup_terminal_outputs(save_path):
    """
    Backup standard and error outputs in terminal to two .txt files named 'stdout.txt' and 'stderr.txt' 
    respectively real time. Terminal would still display outputs as usual.
    Args:
        save_path (directory): directory to save the two .txt files.
    """
    sys.stdout = SysStdLogger(os.path.join(save_path, 'stdout.txt'), sys.stdout)
    sys.stderr = SysStdLogger(os.path.join(save_path, 'stderr.txt'), sys.stderr)		
    
class SysStdLogger(object):
    def __init__(self, filename='terminal log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        self.log.write(''.join([time.strftime("%y-%m-%d %H:%M:%S"), '\n\n']))

    def write(self, message):
        if 'deprecated pixel format used' in message:
            pass
        else:
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.write(''.join(['\n', time.strftime("%y-%m-%d %H:%M:%S")]))
        self.log.close()


def backup_code(save_path=None, ignored_in_current_folder=None, marked_in_parent_folder=None):
    """
    Backup files in current folder and parent folder to "[save_path]/backup_code".
    Also backup standard and error outputs in terminal.
    Args:
        save_path (directory): directory to backup code. If not specified, it would be set to './tmp/[%y%m%d_%H%M%S]'
        ignored_in_current_folder (list): files or folders in this list are ignored when copying files under current folder
        marked_in_parent_folder (list): folders in this list will be copied when copying files under parent folder 
    """
    if ignored_in_current_folder is None:
        ignored_in_current_folder = ['tmp', 'data', '__pycache__']
    if marked_in_parent_folder is None:
        marked_in_parent_folder = ['mylib']

    # set save_path if None
    if save_path==None:
        save_path=os.path.join(sys.path[0], 'tmp', os.path.basename(sys.path[0]), time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # backup terminal outputs
    backup_terminal_outputs(save_path)

    # create directory for backup code
    backup_code_dir = os.path.join(save_path, 'backup_code')
    os.makedirs(backup_code_dir)

    # backup important variables
    with open(os.path.join(backup_code_dir, 'CLI argument.txt'), 'w') as f:
        res = ''.join(['hostName: ', socket.gethostname(), '\n',
                    'account: ', getpass.getuser(), '\n',
                    'save_path: ', os.path.realpath(save_path), '\n', 
                    'CUDA_VISIBLE_DEVICES: ', str(os.environ.get('CUDA_VISIBLE_DEVICES')), '\n'])
        f.write(res)

        for i, _ in enumerate(sys.argv):
            f.write(sys.argv[i] + '\n')

    # copy current script additionally
    script_file = sys.argv[0]
    shutil.copy(script_file, backup_code_dir)

    # copy files in current folder
    current_folder_name = os.path.basename(sys.path[0])
    os.makedirs(os.path.join(backup_code_dir, current_folder_name))
    for file_path in os.listdir(sys.path[0]):
        if file_path not in ignored_in_current_folder:
            if os.path.isdir(file_path):
                shutil.copytree(os.path.join(sys.path[0], file_path), os.path.join(backup_code_dir, current_folder_name, file_path))
            elif os.path.isfile(file_path):
                shutil.copy(os.path.join(sys.path[0], file_path), os.path.join(backup_code_dir, current_folder_name))
            else:
                print('{} is a special file(socket, FIFO, device file) that would not be backup.'.format(file_path))

    # copy folders in parent folder
    os.makedirs(os.path.join(backup_code_dir, 'parent_folder_files'))
    for file_path in os.listdir('../'):
        if os.path.isdir(file_path) and file_path in marked_in_parent_folder:
            shutil.copytree(os.path.join(sys.path[0], '../', file_path), os.path.join(backup_code_dir, file_path))
        elif os.path.isfile(file_path):
            shutil.copy(os.path.join(sys.path[0], '../', file_path), os.path.join(backup_code_dir, 'parent_folder_files'))

# logging functions
class SingleAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count
import time
class AverageMeter(object):
    def __init__(self):
        self.meters = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0        
        self.time_pointer = time.time()
        self.time_meter = SingleAverageMeter()

    def update(self, val, batch_size=1):
        if len(self.meters) < len(val):
            for i in range(len(val)-len(self.meters)):
                self.meters.append(SingleAverageMeter())
        for i, meter in enumerate(self.meters):
            meter.update(val[i],batch_size)
        self.val = [meter.val for meter in self.meters]
        self.avg = [meter.avg for meter in self.meters]
        self.sum = [meter.sum for meter in self.meters]
        self.count = [meter.count for meter in self.meters]
        self.time_meter.update(time.time() - self.time_pointer, batch_size)
        self.time_pointer = time.time()

    @property
    def time_avg(self):
        return self.time_meter.avg

    @property
    def time_val(self):
        return self.time_meter.val
    

import os
from collections import OrderedDict
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None
    
class ResultsLogger():
    """
    A class to log results per epoch/iter in .csv files and tensorboard.
    """
    def __init__(self, save_path, epoch_log_items, train_iter_log_items=None, test_iter_log_items=None, log_in_tensorboard=True):

        # log csv for per epoch
        self.epoch_log_items = epoch_log_items
        self.epoch_log_csv = os.path.join(save_path, 'epoch_logs.csv')
        self.epoch_count = 0
        with open(self.epoch_log_csv, 'w') as f:
            f.write('epoch,')
            for item in epoch_log_items:
                f.write(item+',')
            f.write('\n')

        # log csv for per training iter
        self.train_iter_log_items = train_iter_log_items
        if train_iter_log_items is not None:
            self.train_iter_csv = os.path.join(save_path, 'train_iter_logs.csv')
            self.train_iter_count = 0
            with open(self.train_iter_csv, 'w') as f:
                f.write('iter,')
                for item in self.train_iter_log_items:
                    f.write(item+',')
                f.write('\n')

        # log csv for per test iter
        self.test_iter_log_items = test_iter_log_items
        if test_iter_log_items is not None and log_in_tensorboard:
            self.test_iter_csv = os.path.join(save_path, 'test_iter_logs.csv')
            self.test_iter_count = 0
            with open(self.test_iter_csv, 'w') as f:
                f.write('iter,')
                for item in self.test_iter_log_items:
                    f.write(item+',')
                f.write('\n')
        
        if SummaryWriter is None:
            print('No tensorboardX in this environment. Results won\'t be logged in tensorboard. ')
            log_in_tensorboard = False
        self.log_in_tensorboard = log_in_tensorboard
        if self.log_in_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(save_path, 'Tensorboard_Results'))
        self.save_path = save_path


    def log_epoch(self, results):
        """
        Called per epoch
        """

        # log in logs.csv file
        with open(self.epoch_log_csv, 'a') as f:
            f.write('%d,' % (self.epoch_count,))
            for i, item in enumerate(self.epoch_log_items):
                f.write('%0.6f,' % (results[i]))
            f.write('\n')

        # log in tensorboard
        if self.log_in_tensorboard:
            for i, item in enumerate(self.epoch_log_items):
                self.writer.add_scalar(item, results[i], self.epoch_count)

        self.epoch_count += 1

    def log_train_iter(self, iter_results):
        if self.train_iter_log_items is not None:
            # log in logs.csv file
            with open(self.train_iter_csv, 'a') as f:
                f.write('%d,' % (self.train_iter_count,))
                for i, item in enumerate(self.train_iter_log_items):
                    f.write('%0.6f,' % (iter_results[i]))
                f.write('\n')
                
            # log in tensorboard
            if self.log_in_tensorboard:
                for i, item in enumerate(self.train_iter_log_items):
                    self.writer.add_scalar(item, iter_results[i], self.train_iter_count)
            self.train_iter_count += 1
        else:
            print('No log items for each train iteration.')

    def log_test_iter(self, iter_results):
        if self.test_iter_log_items is not None:
            # log in logs.csv file
            with open(self.test_iter_csv, 'a') as f:
                f.write('%d,' % (self.test_iter_count,))
                for i, item in enumerate(self.test_iter_log_items):
                    f.write('%0.6f,' % (iter_results[i]))
                f.write('\n')
                
            # log in tensorboard
            if self.log_in_tensorboard:
                for i, item in enumerate(self.test_iter_log_items):
                    self.writer.add_scalar(item, iter_results[i], self.test_iter_count)
            self.test_iter_count += 1
        else:
            print('No log items for each test iteration.')

    def close(self, best_result=None):
        if self.log_in_tensorboard:
            self.writer.close()
        if best_result is not None:
            with open(self.epoch_log_csv, 'a') as f:
                f.write('best result, %0.5f\n' % (best_result))

## ----------------------------- end of training utils ----------------------------- ##

