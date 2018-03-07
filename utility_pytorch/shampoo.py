import math
import torch
from torch.optim import Optimizer


class Shampoo(Optimizer):

    def __init__(self, params, lr=1e-1, weight_decay=1e-4):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(Shampoo, self).__init__(params, defaults)

    def quarter(self, mat):
        import IPython
        IPython.embed()
        u, v = torch.symeig(mat, True)
        return v @  (u ** -0.25 + 1.0e-6).diag() @ v.t()

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

                state = self.state[p]

                # State initialization
                if p.dim() == 4:
                    grad = grad.view(p.data.size(0), -1)
                    if len(state) == 0:
                        m, n = grad.size()
                        state['L'] = p.data.new(m, m).zero_() + (p.data.new(m).zero_() + 1.0).diag()
                        state['R'] = p.data.new(n, n).zero_() + (p.data.new(n).zero_() + 1.0).diag()

                    L, R = state['L'], state['R']
                    L = L + grad @ grad.t()
                    R = R + grad.t() @ grad
                    step_size = (group['lr'] * self.quarter(L) @ grad @self.quarter(R)).view(p.data.size())
                else:
                    step_size = group['lr'] * grad
                p.data += -step_size
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)
        return loss
