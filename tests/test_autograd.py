import torch
import pytest
import sys
import os

from source.autograd import ExpPlusCos

def test_autograd_exp_plus_cos():
    x = torch.randn(5, requires_grad=True)
    y = torch.randn(5, requires_grad=True)

    out_custom = ExpPlusCos.apply(x, y)
    out_custom.sum().backward()

    grad_x_custom = x.grad.clone()
    grad_y_custom = y.grad.clone()

    x.grad = None
    y.grad = None

    out_builtin = torch.exp(x) + torch.cos(y)
    out_builtin.sum().backward()

    grad_x_builtin = x.grad
    grad_y_builtin = y.grad

    print("Max diff (x):", (grad_x_custom - grad_x_builtin).abs().max().item())
    print("Max diff (y):", (grad_y_custom - grad_y_builtin).abs().max().item())
