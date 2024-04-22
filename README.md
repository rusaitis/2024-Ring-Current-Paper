# Analysis Software for Rusaitis et al. (2024) Ring Current Paper

![Figure 10](https://github.com/rusaitis/2024-Ring-Current-Paper/figures/figure10.pdf)

## Contents

* `pypic/` - analysis/plotting code
* `figures/` - default figure location
* `DATA/` - default data location
* `Figure[1-13].py` - scripts for the figures
* `configs/` - configuration file for [`iPic3D`](https://github.com/CmPA/iPic3D) (EcSIM).

## Setting up a Virtual Environment

### Conda/Mamba

Simply run `conda env create -f environment.yml` to set up a virtual enivonment called pypic-env.

If using `micromamba`, use `micromamba create -f environment.yml`.

To activate the enironment, run `conda activate pypic-env`.

### PiP

If using the built-in `pip`, simply run `pip install -r requirements.txt`.


## Copy the Data

Place downloaded data from a repository in the `DATA` folder, or provide a custom data path in the code.

## Execute the Code

To execute the code, simply run `python Figure1.py` ...
