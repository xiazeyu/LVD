# LVD Unfolding

A library for performing Latent Variational Diffusion for unfolding variable length collections of objects used in particle phyiscs unfolding.
\
## Installation
You can install this package to use it outside of the repository after cloning.

```bash
git clone https://github.com/Alexanders101/lvd
cd lvd
pip install .
```

Alternatively, you can use `pip install -e .` to install in an editable mode.

## Dependencies

We have updated to using an anaconda environment for simpler dependency management.
You can create the environment locally with the following conda / mamba commands:

```bash
conda env create -p ./environment --file environment_gpu.yaml
conda activate ./environment
pip install .
```

## Example
We have provided a simple `ttbar` example in order to demonstrate how to
define events, construct datasets, and train & evaluate a network.


[Refer to this page for a detailed walk-through 
for the `ttbar` example](docs/TTBar.md).

The full `ttbar` dataset may be downloaded here: http://mlphysics.ics.uci.edu/data/2021_ttbar/.

We also have a more advanced example demonstrating some of the additinoal inputs and outputs available on a semi-leptonic `ttH` event. [Refer to this page for a detailed walk-through 
for the `ttH` example](docs/TTH.md).
