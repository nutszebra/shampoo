import math
import torch
from torch.optim import Optimizer


class Shampoo(Optimizer):

    def __init__(self, params, momentum=0.9, lr=1e-1, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(Shampoo, self).__init__(params, defaults)

    def quarter(self, mat, eps=1.0e-6):
        s, v, d = torch.svd(mat)
        return s @  ((v + eps) ** -0.25).diag() @ s.t()

    @staticmethod
    def prod(size):
        cum = 1
        for i in size:
            cum *= i
        return cum

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                state = self.state[p]
                if len(state) == 0:
                    m, n = grad.size()[0], self.prod(grad.size()[1:])
                    state['step'] = 0
                    state['L'] = p.data.new(m, m).zero_() + (p.data.new(m).zero_() + 1.0e-4).diag()
                    state['R'] = p.data.new(n, n).zero_() + (p.data.new(n).zero_() + 1.0e-4).diag()
                    state['L_inv_quarter'] = p.data.new(m, m).zero_()
                    state['R_inv_quarter'] = p.data.new(n, n).zero_()
                    state['exp_avg'] = grad

                state['exp_avg'].mul_(group['momentum']).add_(1 - group['momentum'], grad)
                grad = state['exp_avg']

                # State initialization
                if p.dim() == 2 or p.dim() == 4:
                    grad = grad.view(p.data.size(0), -1)
                    L, R = state['L'], state['R']
                    L = L + grad @ grad.t()
                    R = R + grad.t() @ grad
                    if (state['step'] % 20) == 0:
                        state['L_inv_quarter'] = self.quarter(L)
                        state['R_inv_quarter'] = self.quarter(R)
                    L_inv_quarter, R_inv_quarter = state['L_inv_quarter'], state['R_inv_quarter']
                    step_size = (group['lr'] * L_inv_quarter @ grad @ R_inv_quarter).view(p.data.size())
                else:
                    # no shampoo on bias or bn
                    step_size = group['lr'] * grad
                p.data += -step_size
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)
                state['step'] += 1
        return loss
