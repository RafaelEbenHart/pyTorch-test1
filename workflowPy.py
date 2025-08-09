import torch
import torch.nn as nn # nn menyediakan layer - layer dari neural network
import matplotlib.pyplot as plt

print(torch.__version__)

# 1. Data (preparing and loading)
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
    plt.scatter(train_data,train_labels, c="b", s=4, label="training data")

    # plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # apakah ada prediksi?
    if predictions is not None:
        # plot predictions jika ada
        plt.scatter(test_data, predictions, c = "r", s= 4, label="Predictions")

    # menampilkan legend
    plt.legend(prop = {"size": 14})

plot_predictions()
plt.show()
