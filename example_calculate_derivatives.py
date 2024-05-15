"""Example python script to calculate the Fisher derivatives with MPI"""
import os
from hd_pk import cmb_from_pk
from hdfisher import mpi

output_dir = os.path.join(os.getcwd(), 'example') # or replace with your path

# decide which kinds of CMB spectra to use:
cmb_types = ['lensed', 'delensed']
# choose the `hd_data_version`; uses HD lensing noise to calculate delensed spectra
hd_data_version = 'latest'

# flags for non-CDM-only models:
baryonic_feedback = False # only used to get default params and step sizes - you can also pass your own
wdm = False # whether to use our WDM transfer function
# flag to include kSZ effect:
ksz = False
# if we're applying a transfer function to the kSZ spectrum, choose the
# fixed redshift at which the transfer function is evaluated; this only
# applied when you include kSZ and either WDM or your own transfer function
ksz_xfer_redshift = 0.5 # the default, for our WDM models

# changing the fiducial parameter values (two ways):
fiducial_params_dict = None
fiducial_params_file = None
# changing the parameter step sizes (two ways):
fisher_steps_dict = None
fisher_steps_file = None
# list of parameters to vary:
fisher_params = None

# arguments for providing your own transfer function:
xfer_dir = None
xfer_params = None

# initialize the `HDFisherFromPk` class:
fisherlib = cmb_from_pk.HDFisherFromPk(output_dir,
        cmb_types=cmb_types,
        hd_data_version=hd_data_version,
        baryonic_feedback=baryonic_feedback,
        wdm=wdm,
        ksz=ksz,
        ksz_xfer_redshift=ksz_xfer_redshift,
        fiducial_params_dict=fiducial_params_dict,
        fiducial_params_file=fiducial_params_file,
        fisher_steps_dict=fisher_steps_dict,
        fisher_steps_file=fisher_steps_file,
        fisher_params=fisher_params,
        xfer_dir=xfer_dir,
        xfer_params=xfer_params)

# calculate the derivatives:
if mpi.rank == 0:
    print('Calculating the derivatives...')
mpi.comm.barrier()

fisherlib.fisher_derivs() # this does the actual calculation
