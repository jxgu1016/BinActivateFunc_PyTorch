#include <torch/torch.h>

#include <vector>

at::Tensor BinActivateFunc_forward(
    at::Tensor input) {
        return at:sign(input);
}

at::Tensor BinActivateFunc_backward(
    at::Tensor input,
    at::Tensor gradinput) {
        return gradinput;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &BinActivateFunc_forward, "BinActivateFunc forward");
  m.def("backward", &BinActivateFunc_backward, "BinActivateFunc backward");
}