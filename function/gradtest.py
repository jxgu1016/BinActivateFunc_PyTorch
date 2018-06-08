import torch
from BinActivateFunc import BinActivateFunc

def main():
    cuda0 = torch.device('cuda:0')
    A = BinActivateFunc.apply
    input = torch.randn(128, 128, 128, requires_grad=True, device=cuda0)
    print(input)
    output = A(input)
    print(output)
    grad_output = torch.randn(128, 128, 128).cuda()
    print(grad_output)
    output.backward(grad_output)
    print(input.grad)
    # manully checking...
    grad_output[input > 1] = 0
    grad_output[input < -1] = 0
    print(torch.equal(grad_output, input.grad))

if __name__ == '__main__':
    main()