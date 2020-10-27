import numpy as np
import time, math, errno
import torch
from torchvision import transforms as tvtf
from torch.utils.data import Dataset
from shutil import copy2
from torch import nn
import pickle, os
from collections import OrderedDict
from torch.optim import lr_scheduler
import matplotlib
import prodict, yaml
import logging

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.init as init

np.set_printoptions(suppress=True, precision=5)


def load_conf(path):
    with open(path, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    return prodict.Prodict.from_dict(yaml_dict)


def timeSince(since, return_seconds=False):
    now = time.time()
    s = now - since
    if return_seconds:
        return s
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def secondSince(since):
    now = time.time()
    s = now - since
    return s


def check_path(path):
    try:
        os.makedirs(path)  # Support multi-level
        print(path + ' created')
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        # print(path, ' exists')


class TrainingProgress:
    def __init__(self, progress_path, result_path, folder_name, tp_step=None, meta_dict=None, record_dict=None,
                 restore=False):
        """
        Header => Filename header,append file name behind
        Data Dict => Appendable data (loss,time,acc....)
        Meta Dict => One time data (config,weight,.....)
        """
        self.progress_path = os.path.join(progress_path, folder_name) + '/'  # RL-tp...
        self.result_path = os.path.join(result_path, folder_name) + '/'
        check_path(self.progress_path)
        check_path(self.result_path)
        if restore:
            assert tp_step is not None, 'Explicitly assign the TP step you want to restore'
            self.restore_progress(tp_step)
        else:
            self.meta_dict = meta_dict or {}  # one time values
            self.record_dict = record_dict or {}  # Recommend, record with step
        self.logger = logging.getLogger('TP')

    def save_model_weight(self, model, epoch, prefix=''):
        name = self.progress_path + prefix + 'model-' + str(epoch) + '.tp'
        torch.save(model.state_dict(), name)

    def restore_model_weight(self, epoch, device, prefix=''):
        name = self.progress_path + prefix + 'model-' + str(epoch) + '.tp'
        return torch.load(name, map_location=device)

    def add_meta(self, new_dict):
        self.meta_dict.update(new_dict)

    def get_meta(self, key):
        try:
            return self.meta_dict[key]
        except KeyError:  # New key
            self.logger.error('TP Error: Cannot find meta, key={}'.format(key))
            return None

    def record_step(self, epoch, prefix, new_dict, display=False):  # use this
        # record every epoch, prefix=train/test/validation....
        key = prefix + str(epoch)
        if key in self.record_dict.keys():
            # print('TP Warning: Epoch Data with key={} is overwritten'.format(key))
            self.record_dict[key].update(new_dict)
        else:
            self.record_dict[key] = new_dict
        if display:
            str_display = ''
            for k, v in new_dict.items():
                if isinstance(v, float):
                    str_display += k + ': {:0.5f}, '.format(v)
                else:
                    str_display += k + ': ' + str(v) + ', '
            self.logger.info(key + ': ' + str_display)

    def get_step_data(self, data_key, prefix, ep_start, ep_end, ep_step=1):
        data = []
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            try:
                data.append(self.record_dict[key][data_key])
            except KeyError:
                self.logger.warning('TP Warning, Invalid epoch={}, Data Ignored!'.format(ep))
        return data

    def get_step_data_all(self, prefix, ep_start, ep_end, ep_step=1):
        ep_end += 1
        data_keys = list(self.record_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = OrderedDict()
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.record_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        return append_dict

    def save_progress(self, tp_step, override_path=None):
        name = self.progress_path + str(tp_step) + '.tpdata' if override_path is None else override_path
        check_path(os.path.dirname(name))
        with open(name, "wb") as f:
            pickle.dump((self.meta_dict, self.record_dict), f, protocol=2)

    def restore_progress(self, tp_step, override_path=None):
        name = self.progress_path + str(tp_step) + '.tpdata' if override_path is None else override_path
        with open(name, 'rb') as f:
            self.meta_dict, self.record_dict = pickle.load(f)

    def plot_data(self, prefix, ep_start, ep_end, file_name, title, ep_step=1, grid=True):  # [ep_start,ep_end]
        ep_end += 1
        data_keys = list(self.record_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = {}
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.record_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        n_cols = 3
        n_rows = int(len(data_keys) / n_cols + 1)
        fig = plt.figure(dpi=800, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(title)
        x_ticks = list(range(ep_start, ep_end, ep_step))
        keys = sorted(append_dict.keys())
        # for i, (k, v) in enumerate(append_dict.items()):
        for i, k in enumerate(keys):
            v = append_dict[k]
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            if grid:
                ax.grid(True)
            ax.plot(x_ticks, v)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_tick_params(labelsize=4)
            ax.set_title(k)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(self.result_path + file_name)
        plt.clf()
        plt.close(fig)

    def plot_data_overlap(self, prefix, ep_start, ep_end, file_name, title, ep_step=1, keys=None):  # [ep_start,ep_end]
        ep_end += 1
        data_keys = list(self.record_dict[prefix + str(ep_start)].keys())
        data_keys.sort()  # Item keys
        append_dict = {}
        for ep in range(ep_start, ep_end, ep_step):
            key = prefix + str(ep)
            for k, v in self.record_dict[key].items():
                try:
                    append_dict[k].append(v)
                except KeyError:
                    append_dict[k] = [v]
        if keys is not None:
            append_dict = {k: append_dict[k] for k in keys}
        fig = plt.figure(dpi=800, figsize=(6, 3))
        fig.suptitle(title)
        x_ticks = list(range(ep_start, ep_end, ep_step))
        keys = sorted(append_dict.keys())
        # for i, (k, v) in enumerate(append_dict.items()):
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        # ax.set_xticks(x_ticks)
        ax.xaxis.set_tick_params(labelsize=4)
        for i, k in enumerate(keys):
            v = append_dict[k]
            # if i == 0:
            #     ax.plot(x_ticks, v, '--', label=k, linewidth=1)
            # else:
            ax.plot(x_ticks, v, label=k, linewidth=1)
        ax.legend()
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(self.result_path + file_name)
        plt.clf()
        plt.close(fig)

    def backup_file(self, src, file_name):  # Saved in result
        self.logger.info('Backup ' + src)
        copy2(src, self.result_path + file_name)

    def save_conf(self, dict, prefix=''):
        path = self.result_path + prefix + 'conf.yaml'
        with open(path, 'w') as outfile:
            yaml.dump(dict, outfile)


class LearningRateScheduler:  # Include torch.optim.lr_scheduler
    def __init__(self, mode, param_groups, lr_rates=None, lr_epochs=None, lr_loss=None, lr_init=None,
                 lr_decay_func=None,
                 torch_lrs='ReduceLROnPlateau', torch_lrs_param={'mode': 'min', 'factor': 0.5, 'patience': 20}):
        self.mode = mode
        if isinstance(param_groups, torch.optim.Optimizer):
            Warning('Deprecated usage, pass list of param group instead')
            self.groups = param_groups.param_groups
        else:
            assert isinstance(param_groups, list)
            self.groups = param_groups  # the specific param group to be controlled
        self.rate = lr_init
        # Check each mode
        if self.mode == 'epoch':
            self.lr_rates = lr_rates  # only single value if decay mode else list of rate
            self.epoch_targets = lr_epochs
            assert (0 <= len(self.lr_rates) - len(self.epoch_targets) <= 1), "Learning rate scheduler setting error."
            self.rate_func = self.lr_rate_epoch
            self.adjust_learning_rate(self.rate)

        elif self.mode == 'loss':
            self.lr_rates = lr_rates
            self.loss_targets = lr_loss
            assert (0 <= len(self.lr_rates) - len(self.loss_targets) <= 1), 'Learning rate scheduler setting error.'
            self.rate_func = self.lr_rate_loss
            self.adjust_learning_rate(self.rate)

        elif self.mode == 'decay':
            self.lr_rates = lr_rates  # only single value if decay mode else list of rate
            self.decay_func = lr_decay_func
            self.rate_func = self.lr_rate_decay
            # raise NotImplementedError  # Zzz....

        elif self.mode == 'torch':
            raise NotImplementedError('TODO: Modify to based on param group')
            # Should set the lr scheduler name in torch.optim.scheduler
            assert torch_lrs_param is not None, "Learning rate scheduler setting error."

            if torch_lrs == 'ReduceLROnPlateau':
                self.torch_lrs = getattr(lr_scheduler, 'ReduceLROnPlateau')(self.optimizer,
                                                                            **torch_lrs_param)  # instance
            else:
                raise NotImplementedError
            self.rate_func = self.torch_lrs.step
        else:
            raise NotImplementedError("Learning rate scheduler setting error.")
        print('Learning rate scheduler: Mode=', self.mode, ' Learning rate=', self.rate)

    def step(self, param_dict, display=True):
        if self.mode == 'torch':
            self.rate_func(param_dict[self.mode])
        else:
            new_rate, self.next = self.rate_func(param_dict[self.mode])
            if new_rate == self.rate:
                return
            else:
                self.rate = new_rate
                if display:
                    print('Learning rate scheduler: Mode=', self.mode, ' New Learning rate=', new_rate,
                          ' Next ', self.mode, ' target=', self.next)
                self.adjust_learning_rate(self.rate)

    def lr_rate_epoch(self, epoch):
        for idx, e in enumerate(self.epoch_targets):
            if epoch < e:
                # next lr rate, next epoch target for changing lr rate
                return self.lr_rates[idx], self.epoch_targets[idx]
        return self.lr_rates[-1], -1  # Last(smallest) lr rate

    def lr_rate_loss(self, loss):
        for idx, l in enumerate(self.loss_targets):
            if loss > l:
                return self.lr_rates[idx], self.loss_targets[idx]  # next lr rate, next loss target for changing lr rate
        return self.lr_rates[-1], -1  # Last(smallest) lr rate

    def lr_rate_decay(self, n):
        rate = self.rate * self.decay_func(n)
        return rate, -1

    def adjust_learning_rate(self, lr):
        for group in self.groups:
            group['lr'] = lr


def initialize_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            print('Conv2d Init')
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            print('BatchNorm Init')
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            print('Linear Init')
            nn.init.xavier_uniform(m.weight)
            nn.init.constant_(m.bias, 1e-3)


def partial_load_weight(dict_src, dict_tgt):
    """
    Example Usage
    >>> dict_src = src_net.state_dict()
    >>> dict_tgt = tgt_net.state_dict()
    >>> dict_tgt = partial_load_weight(dict_src, dict_tgt)
    >>> tgt_net.load_state_dict(dict_tgt)
    """

    keys_src = dict_src.keys()
    for k in dict_tgt.keys():
        if k in keys_src:
            if dict_tgt[k].data.shape == dict_src[k].data.shape:
                dict_tgt[k].data = dict_src[k].data.clone()
                # print(k, ' Loaded')
            else:
                pass
                # print(k, ' Size Mismatched')
    return dict_tgt


class ValueMeter:
    def __init__(self):
        self.data_dict = {}
        self.counter_dict = {}
        self.counter_call = 0

    def record_data(self, dict):
        # assume values are numpy array or python number
        for k, v in dict.items():
            try:
                self.data_dict[k].append(v)
            except KeyError:
                self.data_dict[k] = [v]

    def counter_inc(self, keys):
        for k in keys:
            try:
                self.counter_dict[k] += 1
            except KeyError:
                self.counter_dict[k] = 1
        self.counter_call += 1

    def avg(self):
        result_dict = {}
        for k, v in self.data_dict.items():
            result_dict[k] = np.mean(v)
        return result_dict

    def c_avg(self):  # counter avf
        result_dict = {}
        for k, v in self.counter_dict.items():
            result_dict[k] = v / self.counter_call
        return result_dict

    def std(self):
        result_dict = {}
        for k, v in self.data_dict.items():
            result_dict[k] = np.std(v)
        return result_dict

    def reset(self):
        self.data_dict = {}
        self.counter_dict = {}
        self.counter_call = 0


class ConfNamespace(object):
    def __init__(self, conf_dict, override_dict=None):
        self.__dict__.update(conf_dict)
        if override_dict is not None:
            valid_conf = {k: v for k, v in override_dict.items() if
                          (v is not None) and (v is not False)}
            # Argparse default False if action='store_true'
            self.__dict__.update(valid_conf)


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def train_valid_split(dataset, train_ratio, random_indices=None):
    N = len(dataset)
    train_n = int(train_ratio * N)
    valid_n = N - train_n
    assert train_ratio <= 1
    print('Training set:', train_n, ' , Validation set:', valid_n)
    indices = random_indices if random_indices is not None else np.random.permutation(N)
    assert len(indices) == N
    return Subset(dataset, indices=indices[0:train_n]), Subset(dataset, indices=indices[train_n:N])


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def get_eps_decay(start, final, iter_n):
    return math.exp((math.log(final / start) / iter_n))


class ExplorationRate:  # Epsilon greedy, Prob{eps} random , Prob{1-eps} greedy
    def __init__(self, init_eps, decay_iter, eps_min, eval_eps):
        self.init_eps = init_eps
        self.eps = init_eps
        self.decay = get_eps_decay(init_eps, eps_min, decay_iter)
        self.eps_min = eps_min
        self.iter = 0
        self.eval_eps = eval_eps

    def update(self):
        self.eps = max(self.eps_min, self.eps * self.decay)
        self.iter += 1

    def restore(self, iter):
        self.eps = max(self.eps_min, self.init_eps * (self.decay ** iter))


def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assert img.shape[0] == 3, 'Image is in C,H,W format,float tensor'
    if img.shape[0] == 4:
        img = img[0]
    inv_normalize = tvtf.Normalize(
        mean=np.divide(-np.array(mean), np.array(std)),
        std=1 / np.array(std)
    )
    img = inv_normalize(img)
    img = np.moveaxis(img.numpy(), 0, 2)  # 480,640,3 float
    return img
