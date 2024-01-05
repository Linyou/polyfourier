from setuptools import find_packages, setup

setup(
    name='polyfourier',
    version='0.0.1',    
    description='A PyTorch Polynomial and Fourier series acceleration modules',
    url='https://github.com/Linyou/polyfourier',
    author='Youtian Lin',
    author_email='linyoutian.loyot@gmail.com',
    packages=find_packages(),
    install_requires=[
        'taichi>=1.6',
        'numpy',
        'torch',
    ]
)