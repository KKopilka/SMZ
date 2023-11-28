import torch
import pytest
from Convolution2D import cv2d
import torch.nn as nn

def test_cv2d_1():
    tensor = torch.rand(8, 5, 6)

    Convolution2D = cv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=4, bias=True, padding_mode='zeros')
    result, kernel_size, bias = Convolution2D(tensor)
    torchFunction = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=4, bias=True, padding_mode='zeros')
    torchFunction.weight.data = torch.tensor(kernel_size)
    torchFunction.bias.data = torch.tensor(bias)

def test_cv2d_2():
    tensor = torch.rand(4, 5, 5)

    Convolution2D = cv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, dilation=1, groups=2, bias=True, padding_mode='reflect')
    result, kernel_size, bias = Convolution2D(tensor)
    torchFunction = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, dilation=1, groups=2, bias=True, padding_mode='reflect')
    torchFunction.weight.data = torch.tensor(kernel_size)
    torchFunction.bias.data = torch.tensor(bias)

def test_cv2d_3():
    tensor = torch.rand(2, 2, 2)

    Convolution2D = cv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0, dilation=1, groups=2, bias=True, padding_mode='zeros')
    result, kernel_size, bias = Convolution2D(tensor)
    torchFunction = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0, dilation=1, groups=2, bias=True, padding_mode='zeros')
    torchFunction.weight.data = torch.tensor(kernel_size)
    torchFunction.bias.data = torch.tensor(bias)
