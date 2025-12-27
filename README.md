# Tiny Discrete Diffusion

A simple PyTorch implementation of masked discrete diffusion models for text and images based on the paper [Simplified and Generalized Masked Diffusion for Discrete Data (Shi et al. 2024)](https://arxiv.org/abs/2406.04329).

## Installation

This project uses `uv` to manage a virtual environment with Python 3.11. 

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage
Training and evaluation can be run via the `main.py` script. Example configuration files are provided in the `configs/` directory.

```bash
python main.py --config-name tinyshak \
    use_wandb=true
```

An example script is provided in `scripts/test.sh`.

## TODO

- Add larger datasets and models
- Implement other discrete diffusion variants
- Add multi-GPU training support