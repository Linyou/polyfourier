# A PyTorch Polynomial and Fourier series acceleration modules

This is the DDDM implementation that is been used in the paper: [Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle](https://arxiv.org/abs/2312.03431).

## Installation

```bash
python -m pip install git+https://github.com/Linyou/polyfourier.git
```

## Usage

```python
import torch
import polyfourier

# Initialize taichi
import taichi as ti
ti.init(arch=ti.cuda)

num_points = 100
feature_dim = 3
output_dim = 2

# The parameters should be organized as a tensor of 
# shape (num_points, feature_dim, output_dim)
init_shape = (num_points, feature_dim, output_dim)
params = torch.nn.Parameter(torch.randn(init_shape).requires_grad_()).cuda()
t_array = torch.linspace(0, 1, num_points).reshape(-1, 1).cuda()

# type_name should be 'poly', 'fourier' and 'poly_fourier'
fit_model = polyfourier.get_fit_model(type_name='poly_fourier')

output = fit_model(params, t_array, feature_dim)
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{lin2023gaussian,
  title={Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle},
  author={Lin, Youtian and Dai, Zuozhuo and Zhu, Siyu and Yao, Yao},
  journal={arXiv preprint arXiv:2312.03431},
  year={2023}
}
```
