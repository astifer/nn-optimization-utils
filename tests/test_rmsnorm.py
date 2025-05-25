import torch
from torch.nn import RMSNorm as TorchRMSNorm
from source.rmsnorm import MyRMSNorm

import pytest

def test_check_rmsnorm():
    dim = 16
    x = torch.randn(4, dim, requires_grad=True)

    custom = MyRMSNorm(dim)
    torch_builtin = TorchRMSNorm(dim)

    # Копируем параметры для сравнения
    with torch.no_grad():
        torch_builtin.weight.copy_(custom.scale)

    out_custom = custom(x)
    out_torch = torch_builtin(x)

    print("Max diff:", (out_custom - out_torch).abs().max().item())
