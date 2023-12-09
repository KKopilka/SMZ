import unittest
import torch
import torch.nn as nn
from Convolution2D_dop import cv2d_dop

class TestConvolutionTranspose2D_dop(unittest.TestCase):
    def test_conv_transpose2d_dop_1(self):
        tensor = torch.rand(2, 10, 10)

        ConvTranspose2D = cv2d_dop(in_channels=2, out_channels=2, kernel_size=2, transp_stride=10, bias=True)
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=2, stride=10, bias=True)
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)

    def test_conv_transpose2d_dop_2(self):
        tensor = torch.rand(1, 1, 1)

        ConvTranspose2D = cv2d_dop(in_channels=1, out_channels=1, kernel_size=1, transp_stride=1, bias=True)
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, bias=True)
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)
