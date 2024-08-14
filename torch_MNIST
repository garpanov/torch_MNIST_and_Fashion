import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms
import numpy as np
import time

data_train = torchvision.datasets.MNIST("data/train_data", True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
data_test = torchvision.datasets.MNIST("data/test_data", False, transform=transforms.Compose([transforms.ToTensor()]), download=True)

batch_size = 100

train_load = torch.utils.data.DataLoader(dataset=data_train,
                                         batch_size= batch_size,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(dataset=data_test,
                                        batch_size=1,
                                        shuffle=False)

class TrainModel(nn.Module):
  def __init__(self):
    super(TrainModel, self).__init__()
    self.fx1 = nn.Linear(784,200)
    self.fx2 = nn.Linear(200, 100)
    self.fx3 = nn.Linear(100,10)
    self.m = nn.ReLU()
    self.ant = nn.Dropout()
  def forward(self, x):
    y = self.fx1(self.ant(x))
    y = self.fx2(self.m(y))
    y = self.fx3(y)
    return y

device = torch.device('cpu')

model = TrainModel()
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.01)
a1 = time.time()
for epohs in range(5):
  model.train()
  for label, data in enumerate(train_load):
    X = data[0]
    X = X.view(-1, 28*28)
    Y = data[1]
    pred = model(X)

    loss_model = loss(pred, Y)
    if (label % 200) == 0:
      print(loss_model)
    optimizer.zero_grad()
    loss_model.backward()
    optimizer.step()

print(time.time() - a1)


def accuracy(output, labels):
  predictions = torch.argmax(output, dim=1)
  correct = (predictions == labels).sum().cpu().numpy()
  return correct / len(labels)

rez = torch.tensor([], dtype=torch.float32)
for data_1, data_2 in test_load:
  model.eval()
  X = data_1.view(-1,28*28)
  Y = data_2
  prediction = model(X)
  acc = torch.tensor([accuracy(prediction, Y)], dtype=torch.float32)
  rez = torch.cat((rez, acc), dim=0)

print((rez.sum()/len(rez)))
