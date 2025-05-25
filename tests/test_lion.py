import torch
from torch import nn
from source.lion import Lion  
from matplotlib import pyplot as plt
import numpy as np


def function_to_predict(x, noise=True):
    res = 3 * torch.cos(0.5 * x) 
    if noise:
        res += + torch.rand_like(x) / 100
    return  res

def save_init_function():
    x = torch.tensor(np.linspace(-3, 3, 100))
    y = function_to_predict(x, noise=False)

    plt.plot(x, y, linestyle='-', color='r', label='y=f(x)')
    plt.title("Function to predict")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("other/function_to_predict.png")
    print(f"Function to predict saved into other/function_to_predict.png")

def save_loses_function(loses):
    plt.plot(loses)
    plt.title("Loss plot for Lion oprimizer")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.savefig('other/loss_plot.png')
    print(f"Loss plot saved into other/loss_plot.png")

def test_lion_optimizer():
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    optimizer = Lion(model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    x = torch.randn(256, 1)
    y = function_to_predict(x)

    # print("x size:", x.size())
    # print("y size:", y.size())

    loses = []
    for step in range(1000):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        loses.append(loss.item())

    print(f"Final loss: {loss.item():.4f}")
    save_loses_function()
    plt.clf()
    save_init_function()
