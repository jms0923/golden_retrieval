import numpy as np
import torch
from torch import nn
from torchsummary import summary

from golden_retrieval.descriptors.r_mac import RMAC


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

batch_size = 32
n_classes = 10
n_samples = 1000

# generate random data
X = torch.rand(n_samples, 3, 32, 32)
Y = np.eye(n_classes)[np.random.choice(n_classes, n_samples)]
Y = torch.from_numpy(Y)

### architecture ###
model = nn.Sequential()
model.append(nn.Conv2d(3, 32, 3, padding='same'))
model.append(nn.ReLU())
model.append(nn.Conv2d(32, 32, 3, padding='same'))
model.append(nn.ReLU())
model.append(nn.MaxPool2d(2))
model.append(nn.Dropout(0.25))
model.append(nn.Conv2d(32, 64, 3, padding='same'))
model.append(nn.ReLU())
model.append(nn.MaxPool2d(2))
model.append(nn.Dropout(0.25))

rmac = RMAC(model(X).shape)
model.append(rmac)

model.append(nn.Flatten())
feature_size = model(X).size()

model.append(nn.Linear(feature_size[1], 512))
model.append(nn.ReLU())
model.append(nn.Dropout(0.5))
model.append(nn.Linear(512, n_classes))
model.append(nn.Softmax())

print(model)
summary(model, (3, 32, 32))


### Training ###
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss = 0

model.to(device)
model.train()

for n_batch, (x_batch, y_batch) in enumerate(zip(X, Y)):
    x_batch = x_batch.unsqueeze(0)
    y_batch = y_batch.unsqueeze(0)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    y_pred = model(x_batch)
    loss = loss_func(y_pred, y_batch)
    train_loss += loss.item()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Loss in', n_batch, 'batch :', loss.item())

train_loss = train_loss / n_samples
print('Total Loss: ', train_loss)
