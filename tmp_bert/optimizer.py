from typing import Callable, Iterable, Tuple

import numpy as np
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # group --> {'params': [Parameter containing:
            # tensor([[-0.0043,  0.3097, -0.4752],
            #         [-0.4249, -0.2224,  0.1548]], requires_grad=True)], 'lr': 0.001, 'betas': (0.9, 0.999),
            #         'eps': 1e-06, 'weight_decay': 0.0001, 'correct_bias': True}
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data  # tensor containing the grad of p
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                ''' self looks like this.
                    AdamW (
                    Parameter Group 0
                        betas: (0.9, 0.999)
                        correct_bias: True
                        eps: 1e-06
                        lr: 0.001
                        weight_decay: 0.0001
                    )
                '''
                state = self.state[p]  # {} empty initially
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # exponential moving averages of the gradient (mt)
                    state["mt"] = torch.zeros_like(p.data)
                    # exponential moving averages of the squared gradient (vt)
                    state["vt"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Access hyperparameters from the `group` dictionary
                step_size = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group['eps']

                # Update first and second moments of the gradients
                state['mt'].mul_(beta1)
                state['mt'].add_(grad, alpha=1 - beta1)
                state['vt'].mul_(beta2)
                state['vt'].add_(grad ** 2, alpha=1 - beta2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                m_correct = 1 - beta1 ** state['step']
                v_correct = np.sqrt(1 - beta2 ** state['step'])
                p.data.add_(p.data, alpha=-group['weight_decay'] * step_size)
                p.data.add_(state['mt'] / (torch.sqrt(state['vt']) + eps), alpha=-step_size * v_correct / m_correct)
                # Update parameters

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss
