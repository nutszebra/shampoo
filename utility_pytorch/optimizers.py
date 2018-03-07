import math
import torch.optim as optim
from .utility import write
from . import adamw
from . import shampoo


class FakeOptimizer(object):

    def __call__(self, i):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass

    def info(self):
        pass


class MomentumSGD(object):

    def __init__(self, model, lr, momentum, schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class AdamW(object):

    def __init__(self, model, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.025 * math.sqrt(64 / 50000), t_i=200):
        self.model, self.lr, self.betas = model, lr, betas
        self.eps, self.weight_decay, self.t_i = eps, weight_decay, t_i
        self.optimizer = adamw.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.eta = 1.0
        self.now = {'weight_decay': weight_decay, 'lr': lr}

    @staticmethod
    def cos(t_cur, t_i):
        return 0.5 + 0.5 * math.cos(math.pi * t_cur / t_i)

    def __call__(self, i):
        # updat eta every epoch
        self.eta = self.cos(i, self.t_i)
        for p in self.optimizer.param_groups:
            # lr
            p['lr'] = self.eta * self.lr
            # weight decay
            p['weight_decay'] = self.eta * self.weight_decay * math.sqrt(1.0 / (i + 1))
            self.now['weight_decay'], self.now['lr'] = p['weight_decay'], p['lr']
        self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer: AdamW')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class AMSGradW(object):

    def __init__(self, model, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.025 * math.sqrt(64 / 50000), t_i=200):
        self.model, self.lr, self.betas = model, lr, betas
        self.eps, self.weight_decay, self.t_i = eps, weight_decay, t_i
        self.optimizer = adamw.AMSGradW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.eta = 1.0
        self.now = {'weight_decay': weight_decay, 'lr': lr}

    @staticmethod
    def cos(t_cur, t_i):
        return 0.5 + 0.5 * math.cos(math.pi * t_cur / t_i)

    def __call__(self, i):
        # updat eta every epoch
        self.eta = self.cos(i, self.t_i)
        for p in self.optimizer.param_groups:
            # lr
            p['lr'] = self.eta * self.lr
            # weight decay
            p['weight_decay'] = self.eta * self.weight_decay * math.sqrt(1.0 / (i + 1))
            self.now['weight_decay'], self.now['lr'] = p['weight_decay'], p['lr']
        self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer: AdamW')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class Shampoo(object):

    def __init__(self, model, lr, momentum=0.0, schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = shampoo.Shampoo(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))
