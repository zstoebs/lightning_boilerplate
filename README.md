# Boilerplate for Deep Learning with Pytorch Lightning  

Zach Stoebner

Boilerplate code for building deep learning projects with PyTorch Lightning.  

## Setup

*It is recommended to create a virtual environment for each project.*  

Requirements:  

- python>=3.10
- [pytorch](https://pytorch.org/get-started/locally/)
- [lightning](https://lightning.ai/docs/pytorch/stable/starter/installation.html)

## Usage

The main components are a [dataset](data/__init__.py), a [model]((models/__init__.py)), and a [loss]((losses/__init__.py)). Each module has abstract base classes to define those for a project. Basic classes are provided for each module, as well as an example config and a shell script to easily run configs.  

Either clone the repo or use it as a submodule.  

## References

See [inrlib](https://github.com/utcsilab/inrlib) for more example usage.
