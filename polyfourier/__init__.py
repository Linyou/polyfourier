from typing import Callable

from .poly import Polynomial
from .fourier import Fourier
from .poly_fourier import PolyFourier

def get_fit_model(
    type_name: str = "fourier_poly", 
    feat_dim: int = 3, 
    poly_factor: float = 1.0, 
    Hz_factor: float = 1.0,
) -> Callable:
    if type_name == "fourier":
        trajectory_func = Fourier(
            feat_dim, 
            Hz_base_factor=Hz_factor
        ) 
    elif type_name == "poly_fourier":
        trajectory_func = PolyFourier(
            feat_dim, 
            poly_base_factor=poly_factor,
            Hz_base_factor=Hz_factor
        ) 
    elif type_name == "poly":
        trajectory_func = Polynomial(
            feat_dim,
            poly_base_factor=poly_factor,
        )
    else:
        trajectory_func = None
        print("Trajectory type not found")
    
    return trajectory_func