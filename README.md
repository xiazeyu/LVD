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
We provide a simple dataset for ZJets consitutent unfolding. This dataset is from the following Zenodo: (https://zenodo.org/records/3548091). We provide a tiny sample here to get you gping with the repository and an example structure for the inputs.

`python src/train.py config/zjets.small.main.yaml data/ZJets.Hewig.small.npz`

If you want to provide your own dataset, then the following is the expected structure for the `.npz` file.
```
S: Dataset Size
N: Maximum number of particle-level vectors
M: Maximum number of detector-level vectors

detector_vectors : float : (S, M, DV)   : Detector-level variable-length objects.
detector_event   : float : (S, DE)      : Detector-level event objects (one per sample).
detector_mask    : bool  : (S, M)       : Detector-level variable-length masks (1 = Real, 0 = Padding)
particle_vectors : float : (S, N, PV)   : Particle-level variable-length objects.
particle_event   : float : (S, PE)      : Particle-level event objects (one per sample).
particle_mask    : bool  : (S, N)       : Particle-level variable-length masks (1 = Real, 0 = Padding)
particle_types   : int   : (S, N)       : Particle-level PID values (0 - T, positive only).
```
