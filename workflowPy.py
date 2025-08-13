import torch
import torch.nn as nn # nn menyediakan layer - layer dari neural network
import matplotlib.pyplot as plt
import numpy as np
import pathlib as path  # Digunakan untuk menyimpan model

print(torch.__version__)

## 1. Data (preparing and loading)
# mengubah data menjadi bentuk numerik dan mempelajari representasi (pattern/features/dan weights)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X =torch.arange(start , end, step).unsqueeze(dim=1)
y = weight * X + bias # rumus regresi linear
# x adalah input, y adalah output
print(X, y)
print(len(X), len(y))

## spliting data training dan set test
# 1.training set / wajib digunakan
# 2.validation set / terkadang sering di gunakan
# 3.test set /  wajib digunakan

# membuat train/test split
trainSplit = int (0.8 * len(X))
print(trainSplit)
X_train, y_train = X[:trainSplit], y[:trainSplit] # ambil dari awal sampai indeks trainSplit - 1
X_test, y_test = X[trainSplit:], y[trainSplit:] # ambil dari indeks trainSplit sampai akhir
print(len(X_train), len(y_train), len(X_test), len(y_test))

# visualisasi data
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels= y_test,
                     predictions = None):
    """
    plot training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))

    #  plot training data in blue
    plt.scatter(train_data,train_labels, c="b", s=4, label="Training data")

    # plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # apakah ada prediksi?
    if predictions is not None:
        # plot predictions jika ada
        plt.scatter(test_data, predictions, c = "r", s= 4, label="Predictions")

    # menampilkan legend
    plt.legend(prop = {"size": 14})

plot_predictions()
# plt.show()

## 2. Model (building)

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

## membuat bagaimana model dapat mendekati weight dan bias yang sudah di tentukan
print(f"weight yang di tentukan {weight} dan bias yang di tentukan {bias}")

## membuat prediksi dengan torch.inference_mode()
# mengecek model apakah bisa memprediksi y test dengan x test
# ketika memberikannya pada model deengan menggunakan method forward()

## mengecek hasil dari prediksi model belum di training dengan graph
with torch.inference_mode(): # penggunaan inference_mode adalah untuk mematikan gradient tracking hal ini memberikan keuntungan dari segi kecepatan pemrosesan
    y_pred = model0(X_test)
# atau
with torch.no_grad(): # alternatif lain untuk mematikan gradient tracking
    y_pred = model0(X_test)

print(y_pred)
print(y_test)
plot_predictions(predictions=y_pred)
# plt.show()


## 3. Training & Testing
# membuat model berpindah dari yang random ke data yang sudah ada
# salah satu cara untuk mengukur seberapa buruk model kita adalah dengan menghitung loss
# loss function bisa juga disebut cost function
# loss function - adalah untuk mengukur seberapa buruk model kita dalam memprediksi data
# optimizer - adalah untuk mengupdate parameter model (weight dan bias) agar lebih baik dalam memprediksi data

# spesifik yang di butuhkan
# Training loop
# Testing loop
# MAE - mean absolute error adalah rata-rata dari selisih absolut antara nilai yang diprediksi dan nilai yang sebenarnya

# setup sebelum training
# 1. setup loss function
loss_fn = nn.L1Loss() # L1Loss adalah Mean Absolute Error (MAE)
print(loss_fn)
# 2. setup optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(params = model0.parameters(), lr=0.01) # lr adalah learning rate adalah hyperparameter yang penting dan bisa kamu set dan mempengaruhi hasil training
# lr bisa dilakukan dengan learning rate scheduling

# membuat training loop dan testing loop
# 0. loop ke dalam data
# epoch adalah berapa jumlah iterasi yang akan dilakukan pada seluruh dataset
# epoch adalah hyperparameter
torch.manual_seed(42) # untuk memastikan hasil yang konsisten
epochs = 200

epochCount = []
lossValue = []
testLossValue = []
## untuk mengecek dan membandingkan model dengan model lainnya

for epoch in range(epochs):
    # buat model menadi mode training
    model0.train()  #train mode membuat semua parameter memerlukan gradien
    # 1. foward pass (melibatkan data yang bergerak di model dalam forward function untuk membuat prediction) / foward prop
    y_pred = model0(X_train)
    # 2. menghitung loss (membandingkan prediction dengan target)
    loss = loss_fn(y_pred, y_train) # (input, target)
    # 3. optimizer zero grad
    optimizer.zero_grad() # menghapus gradien dari iterasi sebelumnya agar memudahkan memulai foward pass yang baru dengan lancar
    # 4. loss backward - bergerak mundur melalui jaringan untuk menghitung gradien dari model mempertimbangkan loss (backpropagation)
    loss.backward() # menghitung gradien dari loss terhadap parameter model
    # 5. optimizer step - mengupdate parameter model (weight dan bias) berdasarkan gradien yang telah dihitung (gradient descent)
    optimizer.step() # mengupdate parameter model


    ## Testing
    model0.eval()  # eval mode mematikan fitur model yang di perlukan untuk evaluasi/testing (dropout layers / batch norm)
    # 1. membuat forward pass
    with torch.inference_mode():
        y_predE200 = model0(X_test)
    # 2. menghitung loss
        testLoss = loss_fn(y_predE200, y_test) # menggunakan y test untuk melihat apakah model bisa berkerja dengan data yang tidak diliat sebelumnya

    # print yang telah dikerjakan
    if epoch % 10 == 0: # print setiap 10 epoch
        epochCount.append(epoch)
        lossValue.append(loss.item())
        testLossValue.append(testLoss.item()) ## untuk mengecek dan membandingkan model dengan model lainnya
        testPercent = 100 - testLoss.item() * 100
        lossPercent = 100 - loss.item() * 100
        print(f"Epoch: {epoch}/{epochs} | Loss accuracy: {lossPercent:.2f}% |  Test accuracy: {testPercent:.2f}% | ") # Loss: {loss} | Test loss: {testLoss} |
        print(model0.state_dict())

# kegunaan membagi data set bertujuan agar model dapat di evaluasi dengan data yang tidak terlihat sebelumnya
# dalam hal ini menggunakan y_test

# print(epochCount, lossValue, testLossValue)
# plot loss curves
plt.plot(epochCount, np.array(lossValue), label="Training Loss")
plt.plot(epochCount, testLossValue, label="Testing loss")
plt.title("Training and Testing loss curves")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend()
# plt.show()

## Saving
# method to save the model
# 1. torch.save() - menyimpan object pytorch dalam pickle format
# 2. torch.load() - membuat object pytorch dari pickle format
# 3. torch.nn.Module.load_state_dict() - memuat state dict ke dalam model
# 4. torch.nn.Module.save_state_dict() - menyimpan state dict dari model

# 1. membuat model directory
MODEL_PATH = path.Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. membuat model save path
MODEL_NAME = "LINEAR_REGRESSION_MODEL0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

# 3. menyimpan model state dict
print(f"Saving model state dict to: {MODEL_SAVE_PATH}")
torch.save(obj=model0.state_dict(), f=MODEL_SAVE_PATH)
print(model0.state_dict()) # untuk mengecek apakah model sudah di simpan dengan benar


## Loading
loadedModel0 = LinearRegression()

# 1. Load the saved state_dict of model0
loadedModel0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loadedModel0.state_dict())  # untuk mengecek apakah model sudah di load dengan benar

# 2. membuat prediksi dengan model yang sudah di load
loadedModel0.eval()  # pastikan model dalam mode evaluasi
with torch.inference_mode():
    loadedModel0_pred = loadedModel0(X_test)
print(loadedModel0_pred)

# 3. bandingkan prediksi dengan prediksi model sebelum di load
model0.eval()
with torch.inference_mode():
    y_pred = model0(X_test)
print(y_pred == loadedModel0_pred)