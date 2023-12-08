import unittest
import torch
import torch.nn as nn
from Conv_transpose2d import conv_transpose2d

class TestConvolutionTranspose2D(unittest.TestCase):
    def test_conv_transpose2d_1(self):
        tensor = torch.rand(8, 5, 6)

        ConvTranspose2D = conv_transpose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)

    def test_conv_transpose2d_2(self):
        tensor = torch.rand(4, 5, 5)

        ConvTranspose2D = conv_transpose2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=2, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)

    def test_conv_transpose2d_3(self):
        tensor = torch.rand(2, 2, 2)

        ConvTranspose2D = conv_transpose2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        result, kernel_size, bias = ConvTranspose2D(tensor)
        
        torchFunction = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        torchFunction.weight.data = torch.tensor(kernel_size)
        torchFunction.bias.data = torch.tensor(bias)
