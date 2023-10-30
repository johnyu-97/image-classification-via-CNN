import torch.nn as nn
from torchinfo import summary
import torch

# convolutional layer 1
conv_layer1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5)),
    nn.ReLU(),
)

# convolutional layer 2
conv_layer2 = nn.Sequential(
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
    nn.ReLU(),
)

# convolutional layer 2
conv_layer3 = nn.Sequential(
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3)),
    nn.ReLU(),
)

# fully connected layer 1
fc_layer1 = nn.Sequential(
    nn.Linear(in_features=32*3*3, out_features=64),
    nn.ReLU(),
)

# fully connected layer 2
fc_layer2 = nn.Sequential(
    nn.Linear(in_features=64, out_features=10),
)


LeNet5 = nn.Sequential(
    conv_layer1,
    nn.MaxPool2d(kernel_size=(2,2)),
    conv_layer2,
    nn.MaxPool2d(kernel_size=(2,2)),
    conv_layer3,
    nn.Flatten(), # flatten
    fc_layer1,
    fc_layer2,
)


batch_size = 32
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)

# model.fc = nn.Linear(model.fc.in_features,10)
summary(model, input_size= (batch_size,1,224,224))
# layers[0] = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1)
# features = nn.Sequential(*layers).cuda()