# A PyTorch Polynomial and Fourier series acceleration modules

## Installation

```bash
python -m pip install git+https://github.com/Linyou/poly_fourier.git
```

## Usage

```python
import torch
import polyfourier

num_points = 100
feature_dim = 3
output_dim = 2

# The parameters should be organized as a tensor of 
# shape (num_points, feature_dim, output_dim)
init_shape = (num_points, feature_dim, output_dim)
params = torch.nn.Parameter(torch.randn(init_shape))
t_array = torch.linspace(0, 1, num_points).reshape(-1, 1)

fit_model = polyfourier.get_fit_model(type_name='poly_fourier')

output = fit_model(params, t_array, feature_dim)
```
