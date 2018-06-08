import torch
from torch.autograd import Function
import BinActivateFunc_cpp, BinActivateFunc_cuda

# torch.manual_seed(618)

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

def main():
    cuda0 = torch.device('cuda:0')
    A = BinActivateFunc.apply
    input = torch.randn(4,4, requires_grad=True, device=cuda0)
    print(input)
    output = A(input)
    print(output)
    grad_output = torch.randn(4,4).cuda()
    print(grad_output)
    output.backward(grad_output)
    print(input.grad)

if __name__ == '__main__':
    main()