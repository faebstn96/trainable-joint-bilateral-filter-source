"""
Calculation of gradcheck (PyTorch built-in function to check custom layer implementation)
for verifying a correct gradient implementation.
Documentation: https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html

Author: Fabian Wagner
Contact: fabian.wagner@fau.de
"""
from joint_bilateral_filter_layer import JointBilateralFilterFunction3dCPU, JointBilateralFilterFunction3dGPU
import torch
from torch.autograd import gradcheck


def gradient_input(layer_bf, use_gpu):
    tensor_in = (torch.randn(2, 1, 10, 10, 10, dtype=torch.double, requires_grad=True))
    tensor_guidance = (torch.randn(2, 1, 10, 10, 10, dtype=torch.double))
    if use_gpu:
        tensor_in = tensor_in.cuda()
        tensor_guidance = tensor_guidance.cuda()

    sig_x = torch.tensor(1.1)
    sig_y = torch.tensor(1.1)
    sig_z = torch.tensor(1.1)
    sig_r = torch.tensor(0.5)
    print(gradcheck(layer_bf, (tensor_in, tensor_guidance, sig_x, sig_y, sig_z, sig_r), eps=1e-6, atol=1e-5))


def gradient_guidance(layer_bf, use_gpu):
    tensor_in = (torch.randn(2, 1, 10, 10, 10, dtype=torch.double))
    tensor_guidance = (torch.randn(2, 1, 10, 10, 10, dtype=torch.double, requires_grad=True))
    if use_gpu:
        tensor_in = tensor_in.cuda()
        tensor_guidance = tensor_guidance.cuda()

    sig_x = torch.tensor(1.1)
    sig_y = torch.tensor(1.1)
    sig_z = torch.tensor(1.1)
    sig_r = torch.tensor(0.5)
    print(gradcheck(layer_bf, (tensor_in, tensor_guidance, sig_x, sig_y, sig_z, sig_r), eps=1e-6, atol=1e-5))


def gradient_x(layer_bf, use_gpu):
    tensor_in = (torch.randn(2, 1, 10, 10, 10))
    tensor_guidance = (torch.randn(2, 1, 10, 10, 10))
    if use_gpu:
        tensor_in = tensor_in.cuda()
        tensor_guidance = tensor_guidance.cuda()

    sig_x = torch.tensor(1.1, dtype=torch.double, requires_grad=True)
    sig_y = torch.tensor(1.1)
    sig_z = torch.tensor(1.1)
    sig_r = torch.tensor(0.5)
    print(gradcheck(layer_bf, (tensor_in, tensor_guidance, sig_x, sig_y, sig_z, sig_r), eps=1e-2, atol=1e-3))


def gradient_y(layer_bf, use_gpu):
    tensor_in = (torch.randn(2, 1, 10, 10, 10))
    tensor_guidance = (torch.randn(2, 1, 10, 10, 10))
    if use_gpu:
        tensor_in = tensor_in.cuda()
        tensor_guidance = tensor_guidance.cuda()

    sig_x = torch.tensor(1.1)
    sig_y = torch.tensor(1.1, dtype=torch.double, requires_grad=True)
    sig_z = torch.tensor(1.1)
    sig_r = torch.tensor(0.5)
    print(gradcheck(layer_bf, (tensor_in, tensor_guidance, sig_x, sig_y, sig_z, sig_r), eps=1e-2, atol=1e-3))


def gradient_z(layer_bf, use_gpu):
    tensor_in = (torch.randn(2, 1, 10, 10, 10))
    tensor_guidance = (torch.randn(2, 1, 10, 10, 10))
    if use_gpu:
        tensor_in = tensor_in.cuda()
        tensor_guidance = tensor_guidance.cuda()

    sig_x = torch.tensor(1.1)
    sig_y = torch.tensor(1.1)
    sig_z = torch.tensor(1.1, dtype=torch.double, requires_grad=True)
    sig_r = torch.tensor(0.5)
    print(gradcheck(layer_bf, (tensor_in, tensor_guidance, sig_x, sig_y, sig_z, sig_r), eps=1e-2, atol=1e-3))


def gradient_r(layer_bf, use_gpu):
    tensor_in = (torch.randn(2, 1, 10, 10, 10))
    tensor_guidance = (torch.randn(2, 1, 10, 10, 10))
    if use_gpu:
        tensor_in = tensor_in.cuda()
        tensor_guidance = tensor_guidance.cuda()

    sig_x = torch.tensor(1.1)
    sig_y = torch.tensor(1.1)
    sig_z = torch.tensor(1.1)
    sig_r = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
    print(gradcheck(layer_bf, (tensor_in, tensor_guidance, sig_x, sig_y, sig_z, sig_r), eps=1e-3, atol=1e-3))


# Get BF function.
layer_bf_gpu = JointBilateralFilterFunction3dGPU.apply
layer_bf_cpu = JointBilateralFilterFunction3dCPU.apply

# Note that eps and atol are chosen according to the investigated parameter, as the
# filter parameters operate on different scales/regimes. For example, eps
# (step size for the numerical gradient) must be chosen fairly large for the spatial
# parameters in order to get meaningful gradients. The tolerance parameter atol
# is chosen according to the magnitude of the numerical gradients, determined by eps.

print('-------------------------------------------------------------')
print('Gradcheck is passed if no error occurs and \'True\' is printed.')
print('-------------------------------------------------------------')
print('Gradient with respect to input:')
gradient_input(layer_bf_gpu, use_gpu=True)
gradient_input(layer_bf_cpu, use_gpu=False)

print('Gradient with respect to guidance:')
gradient_guidance(layer_bf_gpu, use_gpu=True)
gradient_guidance(layer_bf_cpu, use_gpu=False)

print('Gradient with respect to sigma_x:')
gradient_x(layer_bf_gpu, use_gpu=True)
gradient_x(layer_bf_cpu, use_gpu=False)

print('Gradient with respect to sigma_y:')
gradient_y(layer_bf_gpu, use_gpu=True)
gradient_y(layer_bf_cpu, use_gpu=False)

print('Gradient with respect to sigma_z:')
gradient_z(layer_bf_gpu, use_gpu=True)
gradient_z(layer_bf_cpu, use_gpu=False)

print('Gradient with respect to sigma_range:')
gradient_r(layer_bf_gpu, use_gpu=True)
gradient_r(layer_bf_cpu, use_gpu=False)
