import unittest
import torch
import torch.nn as nn
from Conv_transpose2d import conv_transpose2d

class TestConvolutionTranspose2D(unittest.TestCase):
    def test_conv_transpose2d_1(self):
        tensor = torch.rand(2, 10, 10)

        ConvTranspose2D = conv_transpose2d(in_channels=2, out_channels=2, kernel_size=2, stride=10, padding=0, output_padding=0, bias=True, dilation=3, padding_mode='zeros')
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=2, stride=10, padding=0, output_padding=0, bias=True, dilation=3, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)

    def test_conv_transpose2d_2(self):
        tensor = torch.rand(1, 1, 1)

        ConvTranspose2D = conv_transpose2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, output_padding=0, bias=True, dilation=3, padding_mode='zeros')
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, output_padding=0, bias=True, dilation=3, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)
