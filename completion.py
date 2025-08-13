import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

# membuat agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. make dataset
weight = 1
bias = 0.5

start = 0
end = 5.0
step = 1.0
X = torch.arange(start, end, step).unsqueeze(dim = 1).to(device)
# print(X)
y = weight * X + bias
print(f"Weight: {weight} | Bias: {bias}")

# 2. split data
trainSplit = int(0.8 * len(X))
# print(trainSplit)
XTrain, yTrain = X[:trainSplit].to(device) , y[:trainSplit].to(device)
XTest, yTest = X[trainSplit:].to(device) , y[trainSplit:].to(device)


# 3. visualisasi Dataset
def plotPredictions(trainData = XTrain.cpu().numpy(),
                     trainLabel = yTrain.cpu().numpy(),
                     testData = XTest.cpu().numpy(),
                     testLabel = yTest.cpu().numpy(),
                    predictions = None):
    """
    Training data,Test data and Prediction
    """
    plt.figure(figsize = (5, 5))
    plt.scatter(trainData, trainLabel, c = "blue", s = 2, label = "Training data")
    plt.scatter(testData, testLabel, c = "green", s = 2, label = "Test Data")
    if predictions is not None:
        plt.scatter(testData, predictions, c = "red", s = 2, label = "predicitons")
    plt.legend(prop = {"size": 10})

# plotPredictions()
# plt.show()

# 4. membuat model
class naik(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearLayer = nn.Linear(in_features= 1, out_features= 1)
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linearLayer(x)

torch.manual_seed(20)
model = naik().to(device)
# print(list(model.parameters()))
print(model.state_dict())

## visualisasi model
with torch.inference_mode():
    yPred = model(XTest)
# plotPredictions(predictions = yPred.cpu().numpy())
# plt.show()

# 4.Training and Testing
# setup
lossFn = nn.L1Loss()
optim = torch.optim.SGD(params = model.parameters(), lr = 0.0211)  #0.0253 / 0.0211

torch.manual_seed(20)
epochs = 200
epochCount = []
lossValue = []
testLossValue = []

# Training

for epoch in range(epochs):
    model.train() # membuat model ke mode train
    yPred = model(XTrain) # melakukan forward pass
    loss = lossFn(yPred, yTrain) # meninjau loss dari input ke target
    optim.zero_grad() # menghapus gradient track dari iterasi sebelum
    loss.backward() # backpropagation
    optim.step() # gradient descent

# Testing
    model.eval() # mode evaluasi
    with torch.inference_mode():
        yPredTrained = model(XTest)
        testLoss = lossFn(yPredTrained, yTest) # menguji model dengan target yang tidak diketahui

    if epoch % 10 == 0:
        epochCount.append(epoch)
        lossValue.append(loss)
        testLossValue.append(testLoss)
        predPercent = 100 - testLoss.item() * 100
        lossPercent = loss * 100
        print(f"Epoch: {epoch}/{epochs} | Model accuracy: {predPercent:.2f}% | Loss: {lossPercent:.2f}% |")

with torch.inference_mode():
    yPred = model(XTest)
plotPredictions(predictions = yPred.cpu().numpy())
plt.show()

plt.plot(epochCount, [i.cpu().detach().numpy() for i in lossValue] ,label = "Training loss") # jika tipe data list maka buat [i.cpu().numpy() for i in target]
plt.plot(epochCount, [i.cpu().detach().numpy() for i in testLossValue], label = "Testing loss")
plt.title("Training and Testing curve")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend()
plt.show()
