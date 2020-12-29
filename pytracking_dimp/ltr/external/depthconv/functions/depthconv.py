import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import cffi
# from .._ext import depthconv

import torch.autograd as ag

try:
    from os.path import join as pjoin, dirname
    from torch.utils.cpp_extension import load as load_extension
    root_dir = pjoin(dirname(__file__), '../src_pytorch13')
    depthconv = load_extension(
        '_depthconv',
        [pjoin(root_dir, 'depthconv_cuda_redo.c'), pjoin(root_dir, 'depthconv_cuda_kernel.cu')],
        verbose=True
    )
except ImportError:
    raise ImportError('Can not compile depth-aware cnn library.')

__all__ = ['depth_conv']

def depth_conv(input,
                  depth,
                  weight,
                  bias,
                  stride=1,
                  padding=0,
                  dilation=1):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = DepthconvFunction(
        _pair(stride), _pair(padding), _pair(dilation))
    # print bias
    if isinstance(bias, torch.nn.Parameter):
        return f(input, depth, weight, bias)
    else:
        return f(input, depth, weight)


import torch.nn as nn
class DepthconvFunction(Function):
    def __init__(self, stride, padding, dilation, bias=True):
        super(DepthconvFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        ffi_=cffi.FFI()
        self.null = ffi_.NULL
        self.bias = bias

    def forward(self, input, depth, weight, bias = None):
        # print('forward')

        if (not self.bias) or (bias is None):
            # print bias, self.bias
            bias = self.null
            bias = input.new(weight.size(0)).zero_()
        input=input.contiguous()
        depth=depth.contiguous()
        weight=weight.contiguous()
        bias  =bias.contiguous()
        self.save_for_backward(input, depth, weight, bias)

        #print(['input.size()', input.size()])
        output_size = [int((input.size()[i + 2] + 2 * self.padding[i] - weight.size()[i + 2]) / self.stride[i] + 1)
                       for i in range(2)]
        #print(['output_size',output_size])
        output = input.new(*self._output_size(input, weight)).zero_()
        self.columns = input.new(weight.size(1) * weight.size(2) * weight.size(3), output_size[0] * output_size[1]).zero_()
        self.ones = input.new(output_size[0] * output_size[1]).zero_()

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            #print([input.size(), depth.size(), weight.size(), bias.size(), output.size()])
            #exit()

            # print([output.size()])
            #print([self.columns.size(), self.ones.size()])
            # [torch.Size([1, 512, 18, 18]), torch.Size([1, 1, 18, 18]), torch.Size([1, 512, 4, 4])]
            # [torch.Size([1, 1, 19, 19])]
            # [torch.Size([8192, 361]), torch.Size([361])]
            # depthconv.depthconv_forward_cuda(
            #         None, None, None, input.new(weight.size(0)).zero_(), None, None, None,
            #         weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0])
            # #print('column', self.columns[10,:])

            depthconv.depthconv_forward_cuda(
                    input, depth, weight, bias, output, self.columns, self.ones,
                    weight.size(3), weight.size(2), self.stride[1], self.stride[0], self.padding[1], self.padding[0], self.dilation[1], self.dilation[0])
            #print('column', self.columns[10,:])
        return output

    def backward(self, grad_output):
        # print('backward')
        # print(self.needs_input_grad)

        input, depth, weight, bias = self.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            if self.needs_input_grad[0]:
                #grad_input = input.new(*input.size()).zero_()
                grad_output = grad_output.contiguous()
                grad_input = torch.zeros_like(input).to(input.device)
                depthconv.depthconv_backward_input_cuda(
                    input, depth, grad_output, grad_input,
                    weight, self.columns,
                    weight.size(3),
                    weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1],
                    self.dilation[0])

            if self.needs_input_grad[2]:


                if len(self.needs_input_grad) == 4:
                    if self.needs_input_grad[3]:
                        grad_bias = weight.new(*bias.size()).zero_()
                    else:
                        grad_bias = self.null
                else:
                    grad_bias = self.null
                #grad_weight = weight.new(*weight.size()).zero_()
                #grad_bias = weight.new(*bias.size()).zero_()
                grad_output = grad_output.contiguous()
                grad_weight=torch.zeros_like(weight).to(weight.device)
                grad_bias  =torch.zeros_like(bias).to(bias.device)
                #print(grad_bias.shape)
                depthconv.depthconv_backward_parameters_cuda(
                    input, depth, grad_output, grad_weight, grad_bias, self.columns,
                    self.ones,
                    weight.size(3),
                    weight.size(2), self.stride[1], self.stride[0],
                    self.padding[1], self.padding[0], self.dilation[1],
                    self.dilation[0], 1)

                if len(self.needs_input_grad) == 4:
                    if not self.needs_input_grad[3]:
                        grad_bias = None
                else:
                    grad_bias = None

        return grad_input, None, grad_weight, grad_bias


    def _output_size(self, input, weight):
        channels = weight.size(0)

        output_size = (input.size(0), channels)

        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
            #print(d, in_size, pad, kernel, stride, output_size)

        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        #print(output_size)
        return output_size
