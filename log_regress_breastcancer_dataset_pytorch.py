import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#0) data preparation

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7)

n_samples, n_features = X.shape

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#model

#f = wx + b  ->  used as input for sigmoid :    1/(1+e**-y) -> results will be between [0,1]

class Model(nn.Module):
    def __init__(self, input_feature_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_feature_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_features)

#loss and optimizer
lr = 0.01

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loop
n_epochs = 10000

for epoch in range(n_epochs):
    optimizer.zero_grad()
    #forward pass
    y_predicted = model(X_train)       #for each x -> y = w*x +b       and afterwards :   y_predicted = 1/(1+e**-y)

    #loss calculation
    loss = criterion(y_predicted, y_train)   # loss is computed according to BCEloss:    -w * [y*log(x) - (1-y)*log(1-x)]  #x != 0 or 1 otherwise log is not def (log(0))

    #backward call to compute local gradients
    loss.backward()                   #dJ/dw    ->            d(1/(1+e**-(w*x + b)) / dw    = - ( x*e**(wx+b))/(1+e**(wx+b)**2)


    #optimizer needs to be updated
    optimizer.step()       #update w with : - (- ( x*e**(wx+b))/(1+e**(wx+b)**2)) *lr         same goes for b with dJ/db

    if (epoch+1) % 1000 == 0:
        print(f"Current epoch: {epoch+1}, loss: {loss.item():.5f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()         #binary outcome but results will be scattered between [0,1] so rounding will make the final call if they are either 1 or 0 class
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy is: {acc.item():.5f}")



