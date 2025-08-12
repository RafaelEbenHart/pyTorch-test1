import torch
import torch.nn as nn
import matplotlib.pyplot as plt

## Membuat class regresi linear
class LinearRegression(nn.Module): # hampir semua dari pytorch adalah "turunan / inheritance" dari nn.module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))

    # foward method to define the computation in the model
    def forward (self, x: torch.tensor) -> torch.tensor: # <- "x" adalah input data
        return self.weights * x + self.bias # rumus regresi linear


## hal esensial dalam pembuatan model pytorch
# 1. torch.nn - menyediakan hal hal yang di perlukan unutk graf komputasi atau neural networks
# 2. torch.nn.parameter - digunakan untuk membuat parameter yang akan di training dan test
# 3. torch.nn.module - kelas dasar untuk semua modul neural network, jika di subclass kan gunakan forward()
# 4. torch.optim - menyediakan algoritma optimasi untuk mengupdate parameter model (optimizer) pada gradient descent
# 5. def forward() - semua nn.module subclass harus mengimplementasikan metode ini

## membuat random seed
torch.manual_seed(42)

# membuat instance dari model
model0 = LinearRegression()

# cek parameter
print(list(model0.parameters()))

# list model dengan nama parameter
print(model0.state_dict())