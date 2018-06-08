import torch
from torch.autograd import Function
import BinActivateFunc_cpp, BinActivateFunc_cuda

torch.manual_seed(618)

class BinActivateFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        ctx.backend = BinActivateFunc_cuda if input.is_cuda else BinActivateFunc_cpp
        output = ctx.backend.forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        ctx.backend.backward(ctx.input, grad_input)
        return grad_input