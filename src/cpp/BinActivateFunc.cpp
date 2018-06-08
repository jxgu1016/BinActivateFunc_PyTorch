#include <torch/torch.h>
#include <vector>
#include <omp.h>

at::Tensor BinActivateFunc_forward(
    at::Tensor input) {
        return at::sign(input);
}

int BinActivateFunc_backward(
    at::Tensor input,
    at::Tensor gradInput) {     
        auto input_data = input.data<float>();
        auto gradInput_data = gradInput.data<float>();
        auto sz = gradInput.numel();
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < sz; i++) {
            if (*(input_data + i) > 1 || *(input_data + i) < -1) {
                *(gradInput_data + i) = 0;
            }
        }
        return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &BinActivateFunc_forward, "BinActivateFunc forward");
  m.def("backward", &BinActivateFunc_backward, "BinActivateFunc backward");
}