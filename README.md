# photon-mosaic-roicat
This is an enxtension for [photon-mosaic](https://github.com/neuroinformatics-unit/photon-mosaic), which also adds support for [ROICaT](https://github.com/RichieHakim/ROICaT). This repository is designed and initiated by Athina Apostolelli.

## Getting started
```sh
conda activate photon-mosaic-dev # We will use the same env

# ROICaT
pip install roicat[all]
pip install git+https://github.com/RichieHakim/roiextractors
pip uninstall torch # because roicat installation will install non-CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# notification
pip install git+https://github.com/neuroinformatics-unit/swc-slack # it works well with any Slack

# Install this module
pip install -e.
```