import torch
import torch.nn as nn
import torch.nn.functional as F   #reLu
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#initialize network

model = NN(input_size=input_size, num_classes=num_classes).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

#train network

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        image = image.reshape(image.shape[0], -1) #keeping first dim and flattening rest into 1 dim , res -> [64, 28*28*1]

        #forward
        scores = model(image)
        loss = criterion(scores, label)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #updates
        optimizer.step()

#check accuracy

def accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            #64x10 (we neeed out of 10 the max) so dim =1)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()      # x/64 max possible
            num_samples += predictions.size(0)  #64 per batch

    acc = 100*(float(num_correct)/float(num_samples))
    print(f"Got {num_correct} / {num_samples} with accuracy {100*float(num_correct)/float(num_samples):.2f}")

    model.train()
    return acc



accuracy(train_loader, model)
accuracy(test_loader, model)



#check accuarcy on training and test to see how good our model works
