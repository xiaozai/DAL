import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from ..functions import depth_conv

__all__ = ['DepthConv']

class DepthConv(Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=1,
                 kernel_size=4,
                 stride=1,
                 padding=2,
                 dilation=1,
                 bias=False):
        super(DepthConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # print('running test_forward')
        # result_test1=self.test_forward()
        # print('test passed, mean and size: ', result_test1.mean(), result_test1.shape)

        # print('running test_backward_input')
        # result_test2=self.test_backward_input()
        # print('test passed, mean and size: ', result_test2.mean(), result_test2.shape)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # n = self.out_channels
        # for k in self.kernel_size:
        #     n *= k
        # self.weight.data.normal_(0, math.sqrt(2. / n))
        # if self.bias is not None:
        #     self.bias.data.zero_()

    def forward(self, input, depth):
        return depth_conv(input, depth, self.weight, self.bias, self.stride,
                             self.padding, self.dilation)
    def test_forward(self):
        #print(feat.shape, filter.shape, depth.shape)
        #torch.Size([1, 512, 18, 18]) torch.Size([1, 512, 4, 4]) torch.Size([1, 1, 18, 18])
        input  = torch.rand((4, 512, 36, 36)).cuda()
        weight = torch.rand((1, 512, 3, 3)).cuda()
        depth  = torch.rand((4, 1, 36, 36)).cuda()
        bias   = torch.rand(weight.size(0)).cuda()
        return depth_conv(input, depth, weight, bias, self.stride,self.padding, self.dilation)
