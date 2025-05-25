import torch

class ExpPlusCos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return torch.exp(x) + torch.cos(y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        dx = torch.exp(x) * grad_output
        dy = -torch.sin(y) * grad_output
        return dx, dy
