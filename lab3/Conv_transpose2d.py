import torch
import torch.nn.functional as F
import numpy as np

def conv_transpose2d(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0, output_padding=0, dilation=1, bias=True, padding_mode='zeros'):
    def convolution_transpose2d(input_m):
        if bias:
            value_b = torch.rand(out_channels)
        else:
            value_b = torch.zeros(out_channels)

        # Проверяем правила для свертки с группами
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        if padding_mode == 'reflect' or padding_mode == 'replicate' or padding_mode == 'circular':
            raise ValueError("Unsupported padding_mode")

        if type(kernel_size) == tuple:
            filter = torch.rand(in_channels, out_channels, kernel_size[0], kernel_size[1])
        elif type(kernel_size) == int:
            filter = torch.rand(in_channels, out_channels, kernel_size, kernel_size)
        else:
            raise ValueError("Unsupported kernel_size type")

        out_tensor = []
        for l in range(out_channels):
            f = np.array([])
            for i in range (0, input_m.shape[1] - ((filter.shape[2]-1) * dilation + 1) + 1, stride):
                for j in range (0, input_m.shape[2] - ((filter.shape[3]-1) * dilation + 1) + 1, stride):
                    s = 0
                    for c in range (in_channels//groups):
                        if groups > 1:
                            val = input_m[l * (in_channels//groups) + c][i:i + (filter.shape[2]-1) * dilation + 1:dilation, j:j + (filter.shape[3]-1) * dilation + 1:dilation]
                        else:
                            val = input_m[c][i:i + (filter.shape[2]-1) * dilation + 1:dilation, j:j + (filter.shape[3] - 1) * dilation + 1:dilation]
                        mini_sum = (val * filter[l][c]).sum()
                        s = s + mini_sum
                    f = np.append(f, float(s + value_b[l]))
            out_tensor.append(torch.tensor(f, dtype=torch.float).view(1, 1, -1))
        return np.array(out_tensor), torch.tensor(np.array(filter)), torch.tensor(np.array(value_b))
    return convolution_transpose2d
