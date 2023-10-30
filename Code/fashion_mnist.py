import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

from torch.utils.data import DataLoader
from dataset import load_fashion_mnist

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

trainset, testset = load_fashion_mnist()
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


## your code here
# TODO: load ResNet18 from PyTorch Hub, and train it to achieve 90+% classification accuracy on Fashion-MNIST.
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# change input feature to 1 as gray scale
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# change output feature to 10 classes
model.fc = nn.Linear(model.fc.in_features,10)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Training Phase
model = model.to(device)        # move model to GPU
# loop over the dataset multiple times
num_epoch = 4
model.train()
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients

        # get prediction
        outputs = model(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:  # print every 500 mini-batches
            print('[epoch %2d, batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

model_file = 'fashion_mnist.pth'
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}.')
print('Finished Training')


# test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

train_acc = accuracy(model, trainloader)
test_acc = accuracy(model, testloader)

print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy on the test set: %f %%' % (100 * test_acc))
