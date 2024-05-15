"""Run the example(s) to calculate the numerical Fisher derivatives"""
import os
import argparse
from hd_pk import cmb_from_pk
from hdfisher import mpi

parser = argparse.ArgumentParser(description="Run the numerical derivatives from the examples in the example Jupyter notebook.")
parser.add_argument('output_dir', help='the absolute path to the directory where you would like the output files to be saved.')
parser.add_argument('example_number', help="the number of the example (e.g. `1`, `2a`) to run.")
parser.add_argument('--xfer_dir', help="(only for example number 4) the absolute path to the directory where the transfer functions are stored.",
                   default=os.path.join(os.getcwd(), 'example_transfer'))
args = parser.parse_args()

output_dir = args.output_dir
xfer_dir = args.xfer_dir
example_number = args.example_number.lower()
example_numbers = ['1', '2a', '2b', '3a', '3b', '3c', '3d', '4']
if example_number not in example_numbers:
    errmsg = f"example number `'{example_number}'` is not recognized. Please choose one of: {example_numbers}."
    raise ValueError(errmsg)

if example_number == '1':
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir)

elif example_number == '2a':
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True)

elif example_number == '2b':
    fid_params, step_sizes = cmb_from_pk.default_fisher_params_step_sizes(baryonic_feedback=False)
    # remove WDM mass
    fid_params.pop('m_wdm', None)
    step_sizes.pop('m_wdm', None)
    # update fiducial value of n_ksz
    fid_params['n_ksz'] = -0.1
    # now run the derivatives
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True, fiducial_params_dict=fid_params, fisher_steps_dict=step_sizes)

elif example_number == '3a':
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True, baryonic_feedback=True)

elif example_number == '3b':
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True, wdm=True, ksz_xfer_redshift=0.5)

elif example_number == '3c':
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True, baryonic_feedback=True, wdm=True, ksz_xfer_redshift=0.5)

elif example_number == '3d':
    fid_params, step_sizes = cmb_from_pk.default_fisher_params_step_sizes(baryonic_feedback=True)
    fid_params['m_wdm'] = 3
    step_sizes['m_wdm'] = 0.1 * fid_params['m_wdm']
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True, baryonic_feedback=True, wdm=True, ksz_xfer_redshift=0.5, fiducial_params_dict=fid_params, fisher_steps_dict=step_sizes)
    
elif example_number == '4':
    wdm_mass = 1
    wdm_step_size = 0.1 * wdm_mass
    xfer_params = ['wdm_mass']
    # use the default parameter and step size dictionaries, and replace `'m_wdm'` with our `'wdm_mass'` parameter:
    fid_params, step_sizes = cmb_from_pk.default_fisher_params_step_sizes(baryonic_feedback=True)
    fid_params.pop('m_wdm', None)
    step_sizes.pop('m_wdm', None)
    fid_params['wdm_mass'] = wdm_mass
    step_sizes['wdm_mass'] = wdm_step_size
    fisherlib = cmb_from_pk.HDFisherFromPk(output_dir, ksz=True, ksz_xfer_redshift=0.5, fiducial_params_dict=fid_params, fisher_steps_dict=step_sizes, xfer_dir=xfer_dir, xfer_params=xfer_params)

  
# do the calculation
if mpi.rank == 0:
    print('calculating the derivatives')
mpi.comm.barrier()
fisherlib.fisher_derivs()

