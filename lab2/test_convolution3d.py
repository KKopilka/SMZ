import unittest
import torch
from Convolution3D import cv3d
import torch.nn as nn

class TestConvolution3D(unittest.TestCase):

    def test_cv3d_1(self):
        tensor = torch.rand(8, 5, 6, 6)

        Convolution3D = cv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=4, bias=True, padding_mode='zeros')
        result, kernel_size, bias = Convolution3D(tensor)
        torchFunction = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=4, bias=True, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)

    def test_cv3d_2(self):
        tensor = torch.rand(4, 5, 5, 5)

        Convolution3D = cv3d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, dilation=1, groups=2, bias=True, padding_mode='reflect')
        result, kernel_size, bias = Convolution3D(tensor)
        torchFunction = nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, dilation=1, groups=2, bias=True, padding_mode='reflect')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)

    def test_cv3d_3(self):
        tensor = torch.rand(2, 2, 2, 2)

        Convolution3D = cv3d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0, dilation=1, groups=2, bias=True, padding_mode='zeros')
        result, kernel_size, bias = Convolution3D(tensor)
        torchFunction = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0, dilation=1, groups=2, bias=True, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)
