# Analysis Software for Rusaitis et al. (2024)

## Contents

* `pypic` - analysis/plotting code
* `figures` - default figure export location
* `DATA` - default data location
* `configs` - configuration file for `iPic3D` (EcSIM). https://github.com/CmPA/iPic3D

## Setting up a Virtual Environment

### Conda/Mamba

Simply run `conda env create -f environment.yml` to set up a virtual enivonment called pypic-env.

If using `micromamba`, use `micromamba create -f environment.yml`.

To activate the enironment, run `conda activate pypic-env`.

### PiP

If using built-in `pip`, simply run `pip install -r requirements.txt`.

conda create -n py39 python=3.9 scipy matplotlib pandas scikit-image colorcet h5py


## Copy the Data

Place downloaded data from a repository in the `DATA` folder, or provide a custom data path in the code.

## Execute the Code

To execute the code, simply run `python Figure1.py` etc.