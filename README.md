# Trainable Joint Bilateral Filter Layer (PyTorch)

This repository implements a GPU-accelerated trainable joint bilateral filter layer (guidance image + three spatial and one range filter dimension) that can be directly included in any Pytorch graph, just as any conventional layer (FCL, CNN, ...). By calculating the analytical derivative of the joint bilateral filter with respect to its parameters, the guidance image, and the input, the (so far) hyperparameters can be automatically optimized via backpropagation for a calculated loss.

Our corresponding paper [Trainable Joint Bilateral Filters for Enhanced Prediction Stability in Low-dose CT](https://arxiv.org/) can be found on [arXiv](https://arxiv.org/) (pre-print).

### Citation:
If you find our code useful, please cite our work
```
@article{wagner2022trainable,
  title={Trainable Joint Bilateral Filters for Enhanced Prediction Stability in Low-dose CT},
  author={Wagner, Fabian and Thies, Mareike and Denzinger, Felix and Gu, Mingxuan and Patwari, Mayank and Ploner, Stefan and Maul, Noah and Pfaff, Laura and Huang, Yixing and Maier, Andreas},
  journal={arXiv preprint arXiv:},
  year={2022},
  doi={https://arxiv.org/}
}
```

### Implementation:

The general structure of the implementation follows the PyTorch documentation for [creating custom C++ and CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html). The forward pass implementation of the layer is based on code from the [Project MONAI](https://docs.monai.io/en/latest/networks.html) framework, originally published under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). The correct implementation of the analytical forward and backward pass can be verified by running the gradcheck.py script, comparing numerical gradients with the derived analytical gradient using the PyTorch built-in [gradcheck function](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html).

### Setup:
The C++/CUDA implemented forward and backward functions are compiled via the setup.py script using setuptools:
1. Create and activate a python environment (python>=3.7).
2. Install [Torch](https://pytorch.org/get-started/locally/) (tested versions: 1.7.1, 1.9.0).
3. Navigate into the extracted repo.
4. Compile/install the bilateral filter layer by calling
```bash
python setup.py install
```

### Example scripts:
* Try out the forward pass by running the example_filter.py (requires [Matplotlib](https://matplotlib.org/stable/users/installing.html) and [scikit-image](https://scikit-image.org/docs/stable/install.html)).
* Run the gradcheck.py script to verify the correct gradient implementation.
* Run example_optimization.py to optimize the parameters of a joint bilateral filter layer to automatically denoise an image.

### Optimized joint bilateral filter prediction:
![](out/example_optimization.png)

### Troubleshooting
#### nvcc-related errors:
1. Compiling the filter layers requires the Nvidia CUDA toolkit. Check its version
   ```bash
   nvcc --version
   ```
   or install it via, e.g.,
   ```bash
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   ```
2. The NVIDIA CUDA toolkit 11.6 made some problems on a Windows machine in combination with pybind. Downgrading the toolkit to version 11.3 fixed the problem (see [this](https://discuss.pytorch.org/t/cuda-11-6-extension-problem/145830) discussion).
#### Windows-related problems:
Make sure the [cl.exe](https://docs.microsoft.com/en-us/cpp/build/reference/compiler-options?view=msvc-170) environment variable is correctly set.