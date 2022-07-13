#include "jointbilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_3d_cpu", &JointBilateralFilterCpuForward, "JBF forward 3d cpu");
    m.def("backward_3d_cpu", &JointBilateralFilterCpuBackward, "JBF backward 3d cpu");
}
