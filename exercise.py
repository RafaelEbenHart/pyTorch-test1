import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pathlib as path
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

weight = 0.3
bias = 0.9

start = 0
end = 1
step = 0.01

X = torch.arange(start, end, step).unsqueeze(dim = 1).to(device)
y = weight * X + bias
# print(len(X))

trainSplit = int(0.8 * len(X))
XTrain , yTrain = X[:trainSplit].to(device) , y[:trainSplit].to(device)
XTest, yTest = X[trainSplit:].to(device), y[trainSplit:].to(device)

def plotPred(trainData = XTrain.cpu().numpy(),
             trainLabel = yTrain.cpu().numpy(),
              testData = XTest.cpu().numpy(),
               testLabel = yTest.cpu().numpy(),
                predictions = None ):
    """
    Train Data,Test Data and Prediction
    """
    plt.figure(figsize = (4, 3))
    plt.scatter(trainData, trainLabel, c = "r", label = "Train data")
    plt.scatter(testData, testLabel, c = "b", label = "Test Data")
    if predictions is not None:
        plt.scatter(testData, predictions, c = "g", label = "predictions")
    plt.legend(prop = {"size" : 10})
plotPred()
# plt.show()

class linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearLayer = nn.Linear(in_features= 1,out_features=1)
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.linearLayer(x)

torch.manual_seed(42)
model = linear().to(device)
print(next(model.parameters()).device)
print(model.state_dict())

with torch.inference_mode():
    yPred = model(XTest)
plotPred(predictions= yPred.cpu().numpy())
# plt.show()

lossFn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.01)

torch.manual_seed(42)
epochs = 300
epochCount = []
lossValue = []
testLossValue = []

for epoch in range(epochs):
    model.train()
    yPred = model(XTrain)
    loss = lossFn(yPred, yTrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        ypredTrained = model(XTest)
        testLoss = lossFn(ypredTrained, yTest)

    if epoch % 20 == 0:
        epochCount.append(epoch)
        lossValue.append(loss)
        testLossValue.append(testLoss)
        trainPer = 100 - testLoss.item() * 100
        lossPer = loss.item() * 100
        print(f"| Epoch: {epoch}/{epochs} | Model Accuracy: {trainPer:.2f}% | Loss : {lossPer:.2f}% |")

with torch.inference_mode():
    ypredTrained = model(XTest)
plotPred(predictions = ypredTrained.cpu().numpy())
# plt.show()

plt.plot(epochCount, [i.cpu().detach().numpy() for i in lossValue], label = "Training Loss")
plt.plot(epochCount, [i.cpu().detach().numpy() for i in testLossValue],label = "Training loss")
plt.title("Training and Testing curve")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend()
# plt.show()

MODEL_PATH = path.Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "LINEAR_REGRESSION_LATIHAN.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

loadedModel = linear().to(device)
loadedModel.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
print(loadedModel.state_dict())

loadedModel.eval()
with torch.inference_mode():
    loadedModelPred = loadedModel(XTest)

model.eval()
with torch.inference_mode():
    yTestPred = model(XTest)
print(yTestPred == loadedModelPred)
