import torch
import numpy as np
import torch.nn as nn
from keras.datasets import fashion_mnist
from torch.utils.data import TensorDataset, DataLoader

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test)

data_train = TensorDataset(x_train, y_train)
data_test = TensorDataset(x_test, y_test)

batch_size = 100
train_load = DataLoader(data_train, batch_size, True)
test_load = DataLoader(data_test, batch_size, False)

class TrainM(nn.Module):
  def __init__(self):
    super(TrainM, self).__init__()
    self.fx1 = nn.Linear(784,200)
    self.fx2 = nn.Linear(200, 150)
    self.fx3 = nn.Linear(150,100)
    self.fx4 = nn.Linear(100, 10)
    self.func = nn.ReLU()
    self.drop = nn.Dropout(0.3)

  def forward(self, x):
    y = self.fx1(self.drop(x))
    y = self.fx2(self.func(y))
    y = self.fx3(self.drop(y))
    y = self.fx4(y)
    return y

device = torch.device('cpu')
model = TrainM()
model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

best_model_weights = None
beast_loss = 30
for epochs in range(20):
  model.train()
  i = 0
  for x, y in train_load:
    x = x.view(-1, 28*28)
    pred = model(x)

    loss_model = loss(pred, y)
    if (i % 200) == 0:
      print(loss_model)
    if beast_loss > loss_model.item():
      beast_loss = loss_model.item()
      best_model_weights = model.state_dict()
    optimizer.zero_grad()
    loss_model.backward()
    optimizer.step()
    i+=1

torch.save(best_model_weights, 'best_model_weights.pth')
model.load_state_dict(torch.load('best_model_weights.pth'))

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
