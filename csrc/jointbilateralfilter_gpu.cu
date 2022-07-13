#include "jointbilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_3d_gpu", &JointBilateralFilterCudaForward, "JBF forward 3d gpu");
    m.def("backward_3d_gpu", &JointBilateralFilterCudaBackward, "JBF backward 3d gpu");
}
