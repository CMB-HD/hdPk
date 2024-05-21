# CMB-HD Matter Power Spectrum and Non-CDM Forecast Code

This repository contains the code used in the forecasts of [MacInnis & Sehgal (2024)](https://arxiv.org/abs/2405.12220). Please cite that work if you use this software or the associated data.

# Installation

## Requirements

To use this software, you must have Python (version >=3) installed, along with the following Python packages:
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [CAMB](https://camb.readthedocs.io/en/latest/)
- [hdfisher](https://github.com/CMB-HD/hdfisher)
- [hdMockData](https://github.com/CMB-HD/hdMockData)
- [matplotlib](https://matplotlib.org/) (this is only required to run the Jupyter notebooks)
- [getdist](https://getdist.readthedocs.io/en/latest/intro.html) (this is only required to run the Jupyter notebooks)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (__optional__: the calculation of the derivatives used in the Fisher matrices can be parallelized, but this is not required.)

## Installation instructions

Simply clone this repository and install with `pip`:

```
git clone https://github.com/CMB-HD/hdPk.git
cd hdPk
pip install . --user
```

# Reproducing the plots and tables in MacInnis & Sehgal (2024)

We provide three Jupyter notebooks that can be run to reproduce the results of MacInnis & Sehgal (2024):
- `hd_lensing_snr.ipynb`: Calculates the CMB-HD lensing SNR and reproduces the associated plot (Figure 6).
- `hd_pk_plot.ipynb`: Reproduces the matter power spectrum plot (Figure 2).
  - Note that many of the data points were originally derived by [Chabanier et. al. (2019)](https://arxiv.org/abs/1905.08103); the code to reproduce the plot in that work can be found at [https://github.com/marius311/mpk_compilation](https://github.com/marius311/mpk_compilation). Here we have added the CMB-HD data points. The notebook can be easily modified to add additional data.
- `hd_non_cdm_forecasts.ipynb`: Reproduces the CMB-HD parameter forecasts for two dark matter models, each with or without baryonic feedback: cold dark matter (CDM-only) or warm dark matter (WDM-only). 


# Running new Fisher forecasts

The Jupyter notebook `examples.ipynb` provides general instructions along with various specific examples of how to run new Fisher forecasts. The python script `run_examples.py` can be used to run the examples in parallel with MPI, instead of inside the `examples.ipynb` notebook; see that notebook for instructions. The python file `example_calculate_derivatives.py` provides a more general "template" you can use as a guide, along with the instructions in `examples.ipynb`. 

We provide the Python code used to calculate the power spectra and Fisher matrices used in MacInnis & Sehgal (2024) for CDM or WDM models, with or without baryonic feedback. By default, we also vary two parameters in our Fisher forecast that characterize the ampltide and slope of the kSZ power spectrum.

You may vary any additional parameters that can be passed to the CAMB `set_params` [function](https://camb.readthedocs.io/en/latest/camb.html#camb.set_params). 

There is also an option to provide your own non-CDM transfer function for the nonlinear matter power spectrum. This can be a single fixed function of wavenumber $k$ and redshift $z$, or it can depend on any number of parameters. In this case, you may also vary the additional parameters that your transfer function depends on (they do not necessarily need to be CAMB parameters). 


