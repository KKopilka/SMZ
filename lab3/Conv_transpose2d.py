import torch
import torch.nn.functional as F
import numpy as np

def conv_transpose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=True, padding_mode='zeros'):
    def convolution_transpose2d(input_m):
        if bias:
            value_b = torch.rand(out_channels)
        else:
            value_b = torch.zeros(out_channels)

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
            f = torch.zeros((input_m.shape[1] - 1) * stride + dilation * (kernel_size - 1) + 1, (input_m.shape[2] - 1) * stride + dilation * (kernel_size - 1) + 1)
            for k in range (in_channels):
                for i in range (0, input_m.shape[1]): 
                    for j in range (0, input_m.shape[2]):
                        val = input_m[k][i][j]
                        proizv = val * filter[k][l]
                        zero_tensor = torch.zeros((filter.shape[2] - 1) * dilation + 1, (filter.shape[3] - 1) * dilation + 1)
                        for t in range (0, zero_tensor.shape[0], dilation):
                            for p in range (0, zero_tensor.shape[1], dilation):
                                zero_tensor[t][p] = proizv[t // dilation][p // dilation]
                        result_tensor = np.add((zero_tensor), f[i * stride:i * stride + (filter.shape[2] - 1) * dilation+1, j * stride:j * stride + (filter.shape[3] - 1) * dilation + 1])
                        f[i * stride:i * stride + (filter.shape[2] - 1) * dilation + 1, j * stride:j * stride + (filter.shape[3] - 1) * dilation + 1] = result_tensor
            out_tensor.append(np.add(f, np.full((f.shape), value_b[l])))
        for q in range(len(out_tensor)):
            if output_padding > 0:
                # Pad along the second dimension (columns)
                out_tensor[q] = torch.cat([torch.zeros(out_tensor[q].shape[0], output_padding), out_tensor[q], torch.zeros(out_tensor[q].shape[0], output_padding)], dim=1)
                # Pad along the first dimension (rows)
                out_tensor[q] = torch.cat([torch.zeros(output_padding, out_tensor[q].shape[1]), out_tensor[q], torch.zeros(output_padding, out_tensor[q].shape[1])], dim=0)
            out_tensor[q] = out_tensor[q][padding:out_tensor[q].shape[0] - padding, padding:out_tensor[q].shape[1] - padding]

        return out_tensor, filter, torch.tensor(value_b)
    return convolution_transpose2d
