import six
import tqdm
import numpy as np
import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from .nutszebra_initialization import Initialization as initializer
except:
    from nutszebra_initialization import Initialization as initializer


class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

    def __setitem__(self, key, value):
        super(NN, self).__setattr__(key, value)
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def _count_parameters(self, shape):
        return functools.reduce(lambda a, b: a * b, shape)

    def count_parameters(self):
        return sum([self._count_parameters(p.data.shape) for p in self.parameters()])

    @staticmethod
    def select_way(way, channel_in, channel_out):
        if way == 'ave':
            n_i = channel_in
            n_i_next = channel_out
        if way == 'forward':
            n_i = channel_in
            n_i_next = None
        if way == 'backward':
            n_i = None
            n_i_next = channel_out
        return n_i, n_i_next

    @staticmethod
    def weight_relu_initialization(link, mean=0.0, relu_a=0.0, way='forward'):
        dim = len(link.weight.data.shape)
        if dim == 2:
            # fc layer
            channel_out, channel_in = link.weight.data.shape
            y_k, x_k = 1, 1
        elif dim == 4:
            # conv layer
            channel_out, channel_in, y_k, x_k = link.weight.data.shape
        n_i, n_i_next = NN.select_way(way, channel_in * y_k * x_k, channel_out * y_k * x_k)
        # calculate variance
        variance = initializer.variance_relu(n_i, n_i_next, a=relu_a)
        # orthogonal matrix
        w = []
        for i in six.moves.range(channel_out):
            w.append(initializer.orthonorm(mean, variance, (channel_in, y_k * x_k), initializer.gauss, np.float32))
        return np.reshape(w, link.weight.data.shape)

    @staticmethod
    def bias_initialization(conv, constant=0.0):
        return initializer.const(conv.bias.data.shape, constant=constant, dtype=np.float32)

    def weight_initialization(self):
        pass

    @staticmethod
    def concatenate_zero_pad(x, pad):
        N, _, H, W = x.data.shape
        x_pad = Variable(torch.zeros(N, pad, H, W), volatile=x.volatile)
        if x.data.type() == 'torch.cuda.FloatTensor':
            x_pad = x_pad.cuda(x.data.get_device())
        x = torch.cat((x, x_pad), 1)
        return x

    def global_average_pooling(self, x):
        batch, ch, height, width = x.data.shape
        x = F.avg_pool2d(x, (height, width), 1, 0)
        return x.view(batch, ch)

    def global_max_pooling(self, x, indices=True):
        batch, ch, height, width = x.data.shape
        if indices is True:
            x, i = F.max_pool2d(x, (height, width), 1, 0, return_indices=indices)
            return x.view(batch, ch), i
        else:
            x = F.max_pool2d(x, (height, width), 1, 0, return_indices=indices)
            return x.view(batch, ch)

    def forward(self, *args, **kwargs):
        return args[0]

    def calc_loss(self, y, t):
        y = F.log_softmax(y)
        loss = F.nll_loss(y, t, weight=None, size_average=True)
        return loss

    @staticmethod
    def print(*args, **kwargs):
        try:
            tqdm.write(*args, **kwargs)
        except:
            print(*args, **kwargs)
