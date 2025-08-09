import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Membuat class regresi linear
class LinearRegression(nn.Module): # hampir semua dari pytorch adalah "turunan / inheritance" dari nn.module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grand = True, dtype = torch.float))

    # foward method to define the computation in the model
    def forward (self, x: torch.tensor) -> torch.tensor: # <- "x" adalah input data
        return self.weights * x + self.bias # rumus regresi linear
