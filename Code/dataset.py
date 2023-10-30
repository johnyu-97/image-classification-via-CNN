import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

preprocess = transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation([0,90]),
                    transforms.ToTensor()
                ])

def load_mnist():
    trainset = MNIST(root='./data', train=True,
                       download=True, transform=ToTensor())
    testset = MNIST(root='./data', train=False,
                      download=True, transform=ToTensor())
    return trainset, testset


def load_cifar10():
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=ToTensor())
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=ToTensor())
    return trainset, testset

def load_fashion_mnist():
    trainset = FashionMNIST(root='./data', train=True,
                            download=True, transform=preprocess)
    testset = FashionMNIST(root='./data', train=False,
                            download=True, transform=preprocess)
    return trainset, testset

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
