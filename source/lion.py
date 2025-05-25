from torch.optim import Optimizer
import torch


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                m = state["momentum"]
                update = (1 - beta1) * grad + beta1 * m
                p.data.add_(-lr * update.sign())

                # Momentum update
                m.mul_(beta2).add_((1 - beta2) * grad)

                # Apply weight decay
                if weight_decay != 0.0:
                    p.data.add_(-lr * weight_decay * p.data)

        return loss
