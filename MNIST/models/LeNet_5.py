from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import time
import math


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        casted = input.type(torch.CharTensor)
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class XNORPopCount(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        pass

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        n = input[0].nelement()
        s = input.size()


class XNORConv2d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
                 Linear=False, previous_conv=False, size=0):

        super(XNORConv2d, self).__init__()
        self.in_channels = in_channels
        self.layer_type = 'XNORConv'
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv
        self.num_filters = out_channels

        self.weight = Parameter(torch.empty((out_channels, in_channels, *(kernel_size, kernel_size))))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def _calc_kernel_alpha(self, n):
        # m = self.weight.data.norm(1, 3).sum(2).sum(1).div(n)

        m = self.weight.data.norm(1, 3, keepdim=True)\
                 .sum(2, keepdim=True)\
                 .sum(1)\
                 .div(n)
        return m

    def _binary_signize_weights(self):
        self.weight = (self.weight.data >= 0).type(torch.uint8)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

        # init.uniform_(self.bias, -bound, bound)
        # for weight in self.parameters():
        #     kaiming_uniform_(weight.data)
        #     #weight.data.uniform_(-1, 1)

    def _get_output_width(self, x):
        w = x.shape[2]
        return ((w - self.kernel_size[0] + 2*self.padding[0])//self.stride[0]) + 1

    def _get_output_height(self, x):
        h = x.shape[3]
        return ((h - self.kernel_size[1] + 2*self.padding[1])//self.stride[1]) + 1

    def _binarized_sign(self, x):
        return (x.sign() >= 0.).type(torch.uint8)

    def forward(self, x):

        batch_size = x.shape[0]
        c, w, h = x.shape[1], x.shape[2], x.shape[3]
        n = c * w * h
        w_new = self._get_output_width(x)
        h_new = self._get_output_height(x)

        alpha = self._calc_kernel_alpha(n)
        kh, kw = self.kernel_size
        N = kh * kw
        dh, dw = self.stride

        # conv = nn.Conv2d(self.in_channels, self.num_filters, (kh, kw), stride=(dh, dw), bias=False)
        # filt = conv.weight.detach()
        # filt.requires_grad = False

        # clamp to -1, 1 for conv; 0, 1 for bitwise
        # conv.weight.data = conv.weight.data.sign()

        # conv.weight.data = (conv.weight.data >= 0.0).type(torch.float)
        # conv.weight.data = (filt >= 0.0).type(torch.float)
        # conv.weight.data[conv.weight.data == 0] = -1.

        bin_filt = self._binarized_sign(self.weight)

        # print(f'slice of binary: {bin_filt[0, 0, :, :]}')

        patches = x.unfold(2, kh, dh).unfold(3, kw, dw).type(torch.int8)
        patches = patches.contiguous().view(batch_size, c, -1, kh, kw)

        nb_windows = patches.size(2)
        patches = patches.permute(0, 2, 1, 3, 4)

        xnor = (~(patches.unsqueeze(2) ^ bin_filt.unsqueeze(0).unsqueeze(1)))
        mask = torch.ones(xnor.shape, device='cuda').type(torch.uint8)
        # anded = (torch.ones(xnor.shape).type(torch.int8) & xnor)
        anded = mask & xnor
        popcount = (2 * anded.sum([5, 4]).type(torch.int8)) - N
        res = popcount.sum([3], dtype=torch.int)

        res = res.permute(0, 2, 1)
        res = res.view(batch_size, -1, h_new, w_new)

        alpha = alpha.expand(self.num_filters, h_new, w_new)
        alpha_v = alpha.view(-1, self.num_filters, h_new, w_new)
        scaled = alpha_v * res
        # x_conv = x.clone().type(torch.float)
        # x_conv[x_conv == 0] = -1.

        # out = conv(x_conv.type(torch.float))
        # print(f'example: {res[0, 0, :, :]} {out[0, 0, :, :]}')
        # print(f'max abs error {(out - res).abs().max()}')

        return scaled # still need to scale by alpha


class BinConv2d(nn.Module): # change the name of BinConv2d
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
                 Linear=False, previous_conv=False, size=0, use_xnor=False):

        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1,
                                     affine=True)
            if use_xnor:
                self.conv = XNORConv2d(input_channels, output_channels,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, groups=groups)
            else:
                self.conv = nn.Conv2d(input_channels, output_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, groups=groups)
        else:
            if self.previous_conv:
                self.bn = nn.BatchNorm2d(int(input_channels/size), eps=1e-4,
                                         momentum=0.1, affine=True)
            else:
                self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1,
                                         affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        # self.alpha = torch.zeros(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)

        x = BinActive.apply(x)

        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            # a_mtrx = torch.mean(x, 1).view((x.shape[0], 1, x.shape[2], x.shape[3]))
            # K = torch.full((a_mtrx.shape[0], 1, self.kernel_size, self.kernel_size), 1./(self.kernel_size ** 2))
            #beta_weights = F.conv2d(a_mtrx, K)
            # print(f'how many zero weights? {(self.conv.weight.data == 0.).sum()}')
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.contiguous().view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x


class LeNet_5(nn.Module):
    def __init__(self, use_xnor=False):
        super(LeNet_5, self).__init__()
        print(f'binary infer? {use_xnor}')
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # added to gradient
        self.bin_conv2 = BinConv2d(20, 50, kernel_size=5, stride=1, padding=0,
                                   use_xnor=use_xnor)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # basically a linear function - added to gradient
        self.bin_ip1 = BinConv2d(50*4*4, 500, Linear=True,
                                 previous_conv=True, size=4*4,
                                 use_xnor=use_xnor)
        self.ip2 = nn.Linear(500, 10)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv1(x)

        # print("BEFORE CONV")
        # print(self.bin_conv2.conv.weight.data[0, 0, :, :])
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)

        # z = self.binary_xnor(x)

        x = self.bin_conv2(x)
        x = self.pool2(x)

        x = self.bin_ip1(x)
        x = self.ip2(x)

        return x


class LeNet_5_vanilla(nn.Module):
    def __init__(self):
        super(LeNet_5_vanilla, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ip1 = nn.Linear(50*4*4, 500)
        self.ip2 = nn.Linear(500, 10)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        return

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu_conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), 50*4*4)

        x = self.ip1(x)
        x = self.ip2(x)
        return x