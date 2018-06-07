import torch
from torch.autograd import Function
import BinActivateFunc_cpp

torch.manual_seed(42)

class BinActivateFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        output = BinActivateFunc_cpp.forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        grad_input = BinActivateFunc_cpp.backward(ctx.input, grad_output)
        return grad_input

def main():
    A = BinActivateFunc.apply
    input = torch.randn(4,4, requires_grad=True)
    print(input)
    output = A(input)
    print(output)
    grad_output = torch.randn(4,4)
    print(grad_output)
    output.backward(grad_output)
    print(input.grad)

if __name__ == '__main__':
    main()