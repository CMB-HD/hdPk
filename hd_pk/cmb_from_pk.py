import os
import warnings
import inspect
import numpy as np
import camb
import yaml
from scipy.interpolate import RectBivariateSpline
from hdfisher import config, utils, theory, fisher, mpi, dataconfig
from hd_mock_data import hd_data


def hd_pk_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_camb_params_list():
    """
    Returns a list of parameter names that can be passed to `camb.set_params()`.
    Note that the list will not contain all possible parameter names that CAMB
    understands, e.g., it won't include all names that are attributes of the
    `camb.CAMBparams` class,  such as `NonLinear` or `Transfer.kmax`.
    """
    camb_param_names = list(camb.get_valid_numerical_params())
    # the following code was taken directly from the `camb.set_params` function:
    cp = camb.model.CAMBparams()
    # list of functions called by `camb.set_params`:
    camb_functions = [cp.set_accuracy, cp.set_classes,
                      cp.DarkEnergy.set_params, cp.set_cosmology, 
                      cp.set_matter_power, cp.set_for_lmax,
                      cp.InitPower.set_params, cp.NonLinearModel.set_params]
    if hasattr(cp.Reion, 'set_extra_params'): # added in version 1.5.1
        camb_functions.append(cp.Reion.set_extra_params)
    for camb_func in camb_functions:
        param_name_list = inspect.getfullargspec(camb_func).args[1:]
        camb_param_names += param_name_list
    camb_param_names = list(set(camb_param_names))
    return camb_param_names


def get_camb_params_dict(params_dict):
    """
    Returns a dictionary holding the names and values in the input
    `params_dict` that can be passed to `camb.set_params`.

    Parameters
    ----------
    params_dict : dict
        A dictionary holding parameter names as the keys and their values.

    Returns
    -------
    camb_params : dict
        A dictionary holding the parameter names and values from the
        `params_dict` that can be passed directly to `camb.set_params`.
    """
    camb_params = {}
    camb_param_names = get_camb_params_list()
    cp = camb.model.CAMBparams()
    for param, value in params_dict.items():
        if param in camb_param_names:
            camb_params[param] = value
        else:
            # the following code was taken directly from `camb.set_params`:
            if '.' in param:
                obj = cp
                parts = param.split('.')
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                if hasattr(obj, parts[-1]):
                    camb_params[param] = value
            else:
                if hasattr(cp, param):
                    camb_params[param] = value
    return camb_params


def read_xfer_data(xfer_fname):
    # read in redshifts from the header:
    with open(xfer_fname, 'r') as f:
        header = f.readline()
    header = header.strip('# \n')
    cols = header.split(', ')
    xfer_zs = []
    for col in cols[1:]:
        col_info = col.split('=')
        z_info = col_info[-1].strip(' ')
        xfer_zs.append(float(z_info))
    xfer_zs = np.array(xfer_zs)
    # get the transfer function at each z and the wavenumbers:
    xfer_data = np.loadtxt(xfer_fname)
    xfer_ks = xfer_data[:,0].copy()
    return xfer_ks, xfer_zs, xfer_data[:,1:]


def interpolate_xfers(xfer_dir, xfer_params=None):
    xfers = {}
    xfer_path = lambda x: os.path.join(xfer_dir, x)
    # for fiducial cosmology:
    xfer_fname = xfer_path('xfer_fid.txt')
    ks, zs, xfer_data = read_xfer_data(xfer_fname)
    xfers['fid'] = RectBivariateSpline(ks, zs, xfer_data, kx=3, ky=3)
    # when each parameter is varied up or down:
    params = [] if xfer_params is None else xfer_params
    for param in params:
        xfers[param] = {}
        for step_dir in ['up', 'down']:
            xfer_fname = xfer_path(f'xfer_{param}_{step_dir}.txt')
            ks, zs, xfer_data = read_xfer_data(xfer_fname)
            xfers[param][step_dir] = RectBivariateSpline(ks, zs, xfer_data, kx=3, ky=3)
    xfer_zmax = zs[-1]
    return xfers, xfer_zmax


def wdm_transfer_lin_viel(k, m_wdm, Omega_wdm, h):
    """k in units of h/Mpc"""
    alpha = 0.049 * (1 / m_wdm)**1.11 * (Omega_wdm/0.25)**0.11 * (h / 0.7)**1.22
    nu = 1.12
    return (1 + (alpha * k)**(2 * nu))**(-10/nu)


def wdm_transfer_nonlin_viel(k, z, m_wdm):
    """k in units of h/Mpc"""
    s = 0.4
    l = 0.6
    nu = 3
    alpha = 0.0476 * (1 / m_wdm)**1.85 * ((1 + z)/2)**1.3
    return (1 + (alpha * k)**(nu * l))**(-s/nu)


def wdm_transfer_nonlin(m_wdm):
    xfer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wdm_transfer_functions/')
    xfer_fname = os.path.join(xfer_dir, f'xfer_wdm_mass{round(m_wdm,2)}keV.txt')
    if not os.path.exists(xfer_fname):
        errmsg = f"Cannot find a precomputed transfer function for {m_wdm} keV WDM."
        raise ValueError(errmsg)
    xfer_ks, xfer_zs, xfer_data = read_xfer_data(xfer_fname)
    return RectBivariateSpline(xfer_ks, xfer_zs, xfer_data, kx=3, ky=3)


def get_wdm_nonlin_transfers(m_wdm, delta_m_wdm=None):
    xfers = {}
    xfers['fid'] = wdm_transfer_nonlin(m_wdm)
    # when varying the mass:
    if delta_m_wdm is not None:
        xfers['m_wdm'] = {}
        step_directions = ['up', 'down']
        masses = [m_wdm + delta_m_wdm, m_wdm - delta_m_wdm]
        for step_dir, mass in zip(step_directions, masses):
            xfers['m_wdm'][step_dir] = wdm_transfer_nonlin(mass)
    return xfers


def wdm_transfer(k, z, m_wdm, Omega_wdm, h):
    transfer_nonlin = wdm_transfer_nonlin(m_wdm)
    if np.ndim(z) > 0: # we have an array of redshifts
        xfer = wdm_transfer_lin_viel(k/h, m_wdm, Omega_wdm, h)
        # replace with non-linear for z <= 14
        loc = np.where(z <= 14)
        xfer[loc] = transfer_nonlin(k[loc], z[loc], grid=False)
    else: # we only have a single value of z
        if z > 14:
            xfer = wdm_transfer_lin_viel(k/h, m_wdm, Omega_wdm, h)
        else:
            xfer = transfer_nonlin(k, z, grid=False)
    return xfer


def calculate_theory_spectra(lmax, camb_params, camb_results=None, 
                             pk_transfer_function=None,
                             cmb_types=['lensed', 'delensed'], 
                             hd_data_version='latest',
                             npts_int=100, zmin=0, zmax=1100, kmax=1000):
    if camb_results is None:
        camb_results = camb.get_results(camb_params)
    # calculate clkk
    clkk = calculate_clkk(camb_params, pk_transfer_function=pk_transfer_function, npts_int=npts_int, zmin=zmin, zmax=zmax, kmax=kmax)
    # calculate and save the CMB spectra
    theo = {}
    for cmb_type in cmb_types:
        theo[cmb_type] = {'ells': np.arange(lmax+1), 'kk': clkk[:lmax+1].copy()}
        if cmb_type == 'delensed': # get residual lensing
            datalib = hd_data.HDMockData(version=hd_data_version)
            _, nlkk = datalib.lensing_noise_spectrum()
            Lmin = datalib.Lmin
            Lmax = datalib.Lmax
            cl = theory.get_residual_lensing(clkk.copy(), nlkk.copy(), Lmin, Lmax, len(clkk)-1)
        elif cmb_type == 'unlensed':
            cl = np.zeros(clkk.shape)
        else:
            cl = clkk.copy()
        theo_spectra = camb_results.get_lensed_cls_with_spectrum(cl * 4 / (2 * np.pi), lmax=lmax, CMB_unit='muK', raw_cl=True)
        for i, s in enumerate(['tt', 'ee', 'bb', 'te']):
            theo[cmb_type][s] = theo_spectra[:,i].copy()
    return theo


def calculate_clkk(camb_params, pk_transfer_function=None, npts_int=100, zmin=0, zmax=1100, kmax=1000):
    """
    pk_transfer_function should be a function of k, z
    """
    PK = camb.get_matter_power_interpolator(camb_params, nonlinear=True,
                                            hubble_units=False, k_hunit=False,
                                            kmax=kmax, zmax=1100,
                                            var1=camb.model.Transfer_Weyl,
                                            var2=camb.model.Transfer_Weyl)
    bg_results = camb.get_background(camb_params)
    chistar = bg_results.conformal_time(0) - bg_results.tau_maxvis
    # integrate P_psi over the given redshift range to get clkk:
    if zmin > 0: # integrate from zmin:
        chi_min = bg_results.comoving_radial_distance(zmin)
    else: # integrate from z = 0 (today):
        chi_min = 0
    if zmax < 1100: # integrate to zmax:
        chi_max = bg_results.comoving_radial_distance(zmax)
    else: # integrate back to recombination:
        chi_max = chistar
    chis = np.linspace(chi_min, chi_max, npts_int)
    zs = bg_results.redshift_at_comoving_radial_distance(chis)
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    # do the integration:
    win = ((chistar - chis) / (chis**2 * chistar))**2
    w = np.ones(chis.shape) # this is just used to set to zero k values out of range of interpolation
    ls = np.arange(2, camb_params.max_l+1, dtype=np.float64)
    cl = np.zeros(camb_params.max_l-1)
    for i, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=1e3] = 0
        integrand = w * PK.P(zs, k, grid=False) * win / k**4
        if pk_transfer_function is not None:
            integrand *= pk_transfer_function(k, zs)
        cl[i] = np.dot(dchis, integrand)
    cl *= (ls*(ls+1))**2
    # return array starting from L = 0:
    clkk = np.zeros(camb_params.max_l+1)
    clkk[2:] = cl
    return clkk


def default_fid_params_fname(baryonic_feedback=False):
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fisher_param_files')
    if baryonic_feedback:
        return os.path.join(param_dir, 'fiducial_params_feedback.yaml')
    else:
        return os.path.join(param_dir, 'fiducial_params.yaml')


def default_step_sizes_fname(baryonic_feedback=False):
    param_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fisher_param_files')
    if baryonic_feedback:
        return os.path.join(param_dir, 'fiducial_step_sizes_feedback.yaml')
    else:
        return os.path.join(param_dir, 'fiducial_step_sizes.yaml')


def default_fisher_params_step_sizes(baryonic_feedback=False):
    param_fname = default_fid_params_fname(baryonic_feedback=baryonic_feedback)
    step_sizes_fname = default_step_sizes_fname(baryonic_feedback=baryonic_feedback)
    fid_params, step_sizes = fisher.get_param_info(param_fname, step_sizes_fname)
    return fid_params, step_sizes


def fid_params(baryonic_feedback=False, only_camb=False):
    param_file = default_fid_params_fname(baryonic_feedback=baryonic_feedback)
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
    if only_camb:
        if 'logA' in params.keys():
            if 'As' not in params.keys():
                params['As'] = np.exp(params['logA']) * 1e-10
            params.pop('logA', None)
        if 'theta' in params.keys():
            if 'cosmomc_theta' not in params.keys():
                params['cosmomc_theta'] = params['theta']
        if 'cosmomc_theta' in params.keys():
            if params['cosmomc_theta'] > 0.1: # make sure we're not passing 100 * theta
                params['cosmomc_theta'] /= 100
        params = get_camb_params_dict(params)
    return params


def load_param_file(param_file):
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_step_sizes(fisher_steps_file, fid_params_dict, varied_param_list=None):
    if varied_param_list is None:
        varied_param_list = [] # so we can still check what's in it
    with open(fisher_steps_file, 'r') as f:
        step_info = yaml.safe_load(f)
    # create a dict of absolute step sizes
    step_sizes = {}
    for param in step_info.keys():
        # make sure we have a fiducial value for that param
        if (param not in fid_params_dict.keys()) and (param in varied_param_list):
            errmsg = (f"There is a step size for {param} in the "
                      "`fisher_steps_file`, but no fiducual value for it. You "
                      "must provide a fiducial value for all varied parameters.")
            raise ValueError(errmsg)
        step = step_info[param]['step_size']
        if 'rel' in step_info[param]['step_type'].lower():
            step *= fid_params_dict[param]
        step_sizes[param] = step
    return step_sizes


def get_fids_step_sizes(fiducial_params_dict=None, fisher_steps_dict=None,
                    fiducial_params_file=None, fisher_steps_file=None,
                    baryonic_feedback=False, varied_param_list=None):
    # get the fiducial parameter values:
    if fiducial_params_dict is not None: # use dict passed by user
        fid_params = fiducial_params_dict.copy()
    else:
        if fiducial_params_file is not None:
            fid_params = load_param_file(fiducial_params_file)
        else:
            default_params_file = default_fid_params_fname(baryonic_feedback=baryonic_feedback)
            fid_params = load_param_file(default_params_file)
    # get the step sizes:
    if fisher_steps_dict is not None:
        step_sizes = fisher_steps_dict.copy()
    else:
        if fisher_steps_file is not None:
            step_sizes = load_step_sizes(fisher_steps_file, fid_params, varied_param_list=varied_param_list)
        else:
            default_step_sizes_file = default_step_sizes_fname(baryonic_feedback=baryonic_feedback)
            step_sizes = load_step_sizes(default_step_sizes_file, fid_params, varied_param_list=varied_param_list)
    return fid_params, step_sizes


def get_varied_params_list(fid_params, step_sizes, varied_param_list=None):
    new_step_sizes = step_sizes.copy()
    # list of varied parameters:
    if varied_param_list is None:
        varied_params = list(new_step_sizes.keys())
    else:
        varied_params = varied_param_list
        all_params = list(new_step_sizes.keys())
        for param in all_params:
            if param not in varied_params:
                new_step_sizes.pop(param, None)
    # for each parameter with a step size, make sure we have a fiducial value:
    missing_fid_params = []
    for param in new_step_sizes.keys():
        if param not in fid_params.keys():
            missing_fid_params.append(param)
    if len(missing_fid_params) > 0:
        errmsg = ("You must provide a fiducial value for the following"
                 f"parameters in order to vary them: {missing_fid_params}")
        raise ValueError(errmsg)
    return varied_params, new_step_sizes



def calc_fisher_matrix(fisher_dir, cmb_types=['lensed', 'delensed'],
                       spectra = ['tt', 'te', 'ee', 'bb', 'kk'],
                       params=None,
                       priors=None,
                       desi_bao=False,
                       hd_data_version='latest'):
    datalib = hd_data.HDMockData(version=hd_data_version)
    lmin = datalib.lmin
    lmax = datalib.lmax
    lmaxTT = datalib.lmaxTT
    bin_edges = datalib.bin_edges()
    if desi_bao:
        # load in precomputed fisher for DESI BAO
        hdfisher_datalib = dataconfig.Data()
        desi_fmat, desi_fisher_params = hdfisher_datalib.load_precomputed_desi_fisher()

    fmats = {}
    fisher_params = {}
    derivs_dir = os.path.join(fisher_dir, 'derivs')
    if not os.path.exists(derivs_dir): # assume user passed the  derivs dir
        derivs_dir = fisher_dir
    # load derivatives up to lmax and calculate fisher matrix for all spectra
    _, fisher_derivs = fisher.load_cmb_fisher_derivs(derivs_dir, cmb_types=cmb_types, lmin=lmin, lmax=lmax, bin_edges=bin_edges)
    # from lmax to lmaxTT
    if 'tt' in spectra:
        _, fisher_derivs_tt = fisher.load_cmb_fisher_derivs(derivs_dir, cmb_types=cmb_types, lmin=lmax, lmax=lmaxTT, bin_edges=bin_edges)
    # get the fisher matrix for each cmb type
    for cmb_type in cmb_types:
        covmat = datalib.block_covmat(cmb_type)
        if params is None:
            fisher_params_list = list(fisher_derivs[cmb_type].keys())
        else:
            fisher_params_list = params
        fmats[cmb_type] = fisher.calc_cmb_fisher(covmat, fisher_derivs[cmb_type], fisher_params_list, spectra=spectra)
        if 'tt' in spectra:
            cov_tt = datalib.tt_diag_covmat(cmb_type)
            if params is None:
                fisher_params_list_tt = list(fisher_derivs_tt[cmb_type].keys())
            else:
                fisher_params_list_tt = params
            fmat_tt = fisher.calc_cmb_fisher(cov_tt, fisher_derivs_tt[cmb_type], fisher_params_list_tt, spectra=['tt'])
            # add the fisher matrices
            fmats[cmb_type], fisher_params[cmb_type] = fisher.add_fishers(fmats[cmb_type], fisher_params_list, fmat_tt, fisher_params_list_tt)
        else:
            fisher_params[cmb_type] = fisher_params_list.copy()
        if desi_bao:
            fmats[cmb_type], fisher_params[cmb_type] = fisher.add_fishers(fmats[cmb_type].copy(), fisher_params[cmb_type],
                                                                          desi_fmat.copy(), desi_fisher_params, priors=priors)
        else:
            # just add the prior(s)
            fmats[cmb_type] = fisher.add_priors(fmats[cmb_type].copy(), fisher_params[cmb_type], priors)
    return fmats, fisher_params



# functions to load pre-computed fisher matrices or theory spectra
def load_precomputed_theory_spectra(cmb_type, m_wdm=None, baryonic_feedback=False):
    # NOTE: no kSZ
    # get the file name:
    theo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'theory_spectra')
    if m_wdm is not None:
        model = f'wdm{int(m_wdm)}keV'
    else:
        model = 'cdm'
    if baryonic_feedback:
        model = f'{model}_baryons'
    fname = os.path.join(theo_dir, f'hd_lmin30lmax40000_{model}_{cmb_type}_cls.txt')
    theo = utils.load_from_file(fname, ['ells', 'tt', 'te', 'ee', 'bb', 'kk'])
    return theo


def load_precomputed_fisher_matrix(cmb_type, m_wdm=None, baryonic_feedback=False, with_ksz=False, with_desi=False):
    # NOTE: all have tau prior of +/- 0.007 applied
    # get the file name:
    fisher_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fisher_matrices')
    fname_root = f'hd_{cmb_type}_fsky0.6_lmin30lmax20100lmaxTT40000'
    if m_wdm is not None:
        model = f'wdm{int(m_wdm)}keV'
    else:
        model = 'cdm'
    if baryonic_feedback:
        model = f'{model}_baryons'
    fname = f'{fname_root}_{model}'
    if with_ksz:
        fname = f'{fname}_ksz'
    if with_desi:
        fname = f'{fname}_desi'
    fname = os.path.join(fisher_dir, f'{fname}_fisher.txt')
    fisher_matrix, fisher_params = fisher.load_fisher_matrix(fname)
    return fisher_matrix, fisher_params



class HDFisherFromPk:
    def __init__(self, output_dir,
                 fiducial_params_dict=None, fisher_steps_dict=None,
                 fiducial_params_file=None, fisher_steps_file=None,
                 baryonic_feedback=False,
                 fisher_params=None,
                 wdm=False,
                 xfer_dir=None, xfer_params=None,
                 ksz=False, ksz_xfer_redshift=0.5,
                 cmb_types=['lensed', 'delensed'],
                 hd_data_version='latest'): 
        # setup to calculate theory
        self.hd_data_version = hd_data_version
        self.datalib = hd_data.HDMockData(version=self.hd_data_version)
        self.lmin = self.datalib.lmin
        self.lmax = self.datalib.lmaxTT
        self.Lmin = self.datalib.Lmin
        self.Lmax = self.datalib.Lmax
        self.ells = np.arange(self.lmax+1)
        _, self.nlkk = self.datalib.lensing_noise_spectrum()
        self.spectra = ['tt', 'te', 'ee', 'bb', 'kk']
        self.theo_cols = ['ells'] + self.spectra
        self.npts_int = 100 # number of points to use when integrating clkk
        self.cmb_types = cmb_types

        self.ksz = ksz

        # transfer function for non-CDM models:
        # either our precomputed WDM transfer function,
        self.wdm = wdm
        self.wdm_xfer_zmax = 14
        # or interpolate it from a file,
        self.xfer = (xfer_dir is not None)
        if self.xfer and self.wdm:
            errmsg = ("`wdm` must be `False` if `xfer_dir` is not `None`; "
                      "You can either use the precomputed WDM transfer "
                      "function by setting `wdm = True`, or use your own "
                      "transfer function by passing the `xfer_dir`.")
            raise ValueError(errmsg)


        # get list of parameters to vary and their step sizes:
        self.fid_params, step_sizes = self.init_params(fiducial_params_dict=fiducial_params_dict,
                                                        fisher_steps_dict=fisher_steps_dict,
                                                        fiducial_params_file=fiducial_params_file,
                                                        fisher_steps_file=fisher_steps_file,
                                                        baryonic_feedback=baryonic_feedback,
                                                        fisher_params=fisher_params,
                                                        xfer_params=xfer_params)
        self.fisher_params, self.step_sizes = get_varied_params_list(self.fid_params, step_sizes,
                                                                     varied_param_list=fisher_params)

        # need h and dark matter density for the WDM transfer function:
        self.h, self.Omega_dm = self.get_OmegaDM_and_h(self.fid_params)
        # list of tuples of param (name, value)
        self.param_values = fisher.get_varied_param_values(self.fid_params, self.step_sizes)

        # load the transfer functions
        if self.xfer:
            if xfer_params is not None:
                self.xfer_params = xfer_params.copy()
            else:
                self.xfer_params = []
            self.xfer_dict, self.xfer_zmax = interpolate_xfers(xfer_dir, xfer_params=xfer_params)
        elif self.wdm:
            self.xfer_params = ['m_wdm']
            if 'm_wdm' in self.step_sizes.keys():
                delta_m_wdm = self.step_sizes['m_wdm']
            else:
                delta_m_wdm = None
            self.wdm_nonlin_transfer_funcs = get_wdm_nonlin_transfers(self.fid_params['m_wdm'], delta_m_wdm=delta_m_wdm)
        else:
            self.xfer_params = []

        self.ksz_ells, self.cl_ksz = self.datalib.cl_ksz()
        if self.wdm or self.xfer:
            self.ksz_xfer_redshift = ksz_xfer_redshift
            if self.ksz_xfer_redshift is not None:
                # get wavenumber corresponding to each multipole at this redshift:
                pars = self.set_camb_params()
                bg_results = camb.get_background(pars)
                chi = bg_results.comoving_radial_distance(self.ksz_xfer_redshift)
                self.ks_xfer = (self.ells + 0.5) / chi
                self.zs_xfer = np.array([self.ksz_xfer_redshift] * len(self.ells))
        else:
            self.ksz_xfer_redshift = None

        # distribute the calculations over the MPI processes
        ntasks = len(self.param_values)
        self.task_idxs = mpi.distribute(ntasks, mpi.size, mpi.rank)
        self.my_ntasks = len(self.task_idxs) # for this MPI process

        # create directories where output is saved:
        self.fisher_dir = output_dir
        self.theo_dir = os.path.join(self.fisher_dir, 'theory')
        self.derivs_dir = os.path.join(self.fisher_dir, 'derivs')
        if mpi.rank == 0:
            utils.set_dir(self.fisher_dir)
            utils.set_dir(self.theo_dir)
            utils.set_dir(self.derivs_dir)
        mpi.comm.barrier()


    def init_params(self, fiducial_params_dict=None, fisher_steps_dict=None, 
                    fiducial_params_file=None, fisher_steps_file=None, 
                    baryonic_feedback=False, fisher_params=None, xfer_params=None):
        if xfer_params is None:
            xfer_params = [] # so we can still look for items in list
        fid_params, step_sizes = get_fids_step_sizes(fiducial_params_dict=fiducial_params_dict, 
                                                     fisher_steps_dict=fisher_steps_dict, 
                                                     fiducial_params_file=fiducial_params_file, 
                                                     fisher_steps_file=fisher_steps_file, 
                                                     baryonic_feedback=baryonic_feedback, 
                                                     varied_param_list=fisher_params)
        # the default parameters and step sizes include kSZ and WDM params;
        # remove them if they're not needed:
        using_default_params = (fiducial_params_dict is None) and (fiducial_params_file is None)
        if using_default_params:
            if (not self.wdm) and ('m_wdm' not in xfer_params):
                fid_params.pop('m_wdm', None)
                step_sizes.pop('m_wdm', None)
            if (not self.ksz) and ('A_ksz' not in xfer_params):
                fid_params.pop('A_ksz', None)
                step_sizes.pop('A_ksz', None)
            if (not self.ksz) and ('n_ksz' not in xfer_params):
                fid_params.pop('n_ksz', None)
                step_sizes.pop('n_ksz', None)
        # alternatively, if the user passed their own parameters, make sure
        # that we have kSZ and/or WDM params, if they're needed:
        else:
            param_names = list(fid_params.keys())
            if self.ksz:
                if ('A_ksz' not in param_names) or ('n_ksz' not in param_names):
                    errmsg = ("To include the kSZ effect, you must provide "
                              "fiducial values for both `'A_ksz'` and `'n_ksz'` "
                              "in either the `fiducial_params_dict` "
                              "or the `fiducial_params_file`.")
                    raise ValueError(errmsg)
            if self.wdm:
                if 'm_wdm' not in param_names:
                    errmsg = ("To use a precomputed WDM transfer function, you "
                              "must provide a fiducial value for `'m_wdm'` in "
                              "either the `fiducial_params_dict` "
                              "or the `fiducial_params_file`.")
                    raise ValueError(errmsg)
        return fid_params, step_sizes


    def get_param_dicts(self, varied_param_name=None, varied_param_value=None):
        param_dict = self.fid_params.copy()
        param_dict['lmax'] = int(self.lmax+500)
        if varied_param_name is not None:
            param_dict[varied_param_name] = varied_param_value
        # deal with logA vs. As
        if ('logA' in param_dict):
            if varied_param_name != 'As': # make sure we use the new value for As
                param_dict['As'] = np.exp(param_dict['logA'])/1.e10
            param_dict.pop('logA', None)
        # if the `varied_param_name` is `'theta'` or `'cosmomc_theta`', make sure we use that value
        # and also make sure we have theta, not 100*theta, to pass to camb
        if varied_param_name is not None:
            if 'theta' in varied_param_name: # make sure we use the correct value
                param_dict['cosmomc_theta'] = varied_param_value
        if 'theta' in param_dict.keys(): # CAMB calls it 'cosmomc_theta'
            if 'cosmomc_theta' not in param_dict.keys():
                param_dict['cosmomc_theta'] = param_dict['theta']
            param_dict.pop('theta', None)
        if 'cosmomc_theta' in param_dict.keys():
            if param_dict['cosmomc_theta'] > 0.1: # make sure we're not passing 100 * theta
                param_dict['cosmomc_theta'] /= 100
        camb_params = get_camb_params_dict(param_dict)
        return param_dict, camb_params


    def set_camb_params(self, varied_param_name=None, varied_param_value=None):
        _, camb_params = self.get_param_dicts(varied_param_name=varied_param_name, varied_param_value=varied_param_value)
        pars = camb.set_params(**camb_params)
        return pars


    def get_OmegaDM_and_h(self, params):
        h = None
        Omega_dm = None
        # if the params are already in the dict, just return them
        if 'H0' in params.keys():
            h = params['H0'] / 100
            if 'omch2' in params.keys():
                Omega_dm = params['omch2'] / h**2
                return h, Omega_dm
        # otherwise, calculate them from the parameters that were passed
        pars = self.set_camb_params()
        bg_results = camb.get_background(pars)
        if Omega_dm is None:
            densities = bg_results.get_background_densities(1.0, vars=['tot', 'cdm'], format='dict')
            Omega_dm = densities['cdm'] / densities['tot']
        if h is None:
            h = bg_results.hubble_parameter(0) / 100
        return h, Omega_dm


    def wdm_transfer(self, k, z, m_wdm):
        # get the linear transfer function:
        xfer = wdm_transfer_lin_viel(k/self.h, m_wdm, self.Omega_dm, self.h)
        # replace with non-linear for z <= 14:
        loc = np.where(z <= self.wdm_xfer_zmax)
        if m_wdm > self.fid_params['m_wdm']:
            xfer[loc] = self.wdm_nonlin_transfer_funcs['m_wdm']['up'](k[loc], z[loc], grid=False)
        elif m_wdm < self.fid_params['m_wdm']:
            xfer[loc] = self.wdm_nonlin_transfer_funcs['m_wdm']['down'](k[loc], z[loc], grid=False)
        else:
            xfer[loc] = self.wdm_nonlin_transfer_funcs['fid'](k[loc], z[loc], grid=False)
        return xfer


    # TODO: test and remove`
    """
    def pk_transfer(self, k, z, varied_param_name=None, varied_param_value=None):
        # assume transfer function is one above a certain redshift
        xfer = np.ones(len(k))
        # get the transfer function below this redshift
        loc = np.where(z <= self.xfer_zmax)
        if varied_param_name in self.xfer_params:
            step_dir = 'up' if (varied_param_value > self.fid_params[varied_param_name]) else 'down'
            xfer[loc] = self.xfer_dict[varied_param_name][step_dir](k[loc], z[loc], grid=False)
        else:
            xfer[loc] = self.xfer_dict['fid'](k[loc], z[loc], grid=False)
        return xfer
    """
    def pk_transfer(self, k, z, varied_param_name=None, varied_param_value=None):
        # get the transfer function up to the zmax of the stored transfer function;
        # above that, assume that the transfer function about zmax is independent of z,
        #  and use the transfer function evaluated at zmax:
        loc = np.where(z <= self.xfer_zmax)
        if varied_param_name in self.xfer_params:
            step_dir = 'up' if (varied_param_value > self.fid_params[varied_param_name]) else 'down'
            # first get the transfer function above zmax:
            xfer = self.xfer_dict[varied_param_name][step_dir](k, self.xfer_zmax, grid=False)
            # then replace it  with the redshift-dependent function below zmax:
            xfer[loc] = self.xfer_dict[varied_param_name][step_dir](k[loc], z[loc], grid=False)
        else:
            # first get the transfer function above zmax:
            xfer = self.xfer_dict['fid'](k, self.xfer_zmax, grid=False)
            # then replace it with the redshift-dependent function below zmax:
            xfer[loc] = self.xfer_dict['fid'](k[loc], z[loc], grid=False)
        return xfer



    def calculate_cl_ksz(self, varied_param_name=None, varied_param_value=None):
        if varied_param_name == 'A_ksz':
            A_ksz = varied_param_value
        else:
            A_ksz = self.fid_params['A_ksz']
        if varied_param_name == 'n_ksz':
            n_ksz = varied_param_value
        else:
            n_ksz = self.fid_params['n_ksz']
        if self.wdm and (self.ksz_xfer_redshift is not None):
            if varied_param_name == 'm_wdm':
                m_wdm = varied_param_value
            else:
                m_wdm = self.fid_params['m_wdm']
            xfer = self.wdm_transfer(self.ks_xfer, self.zs_xfer, m_wdm)
        elif self.xfer and (self.ksz_xfer_redshift is not None):
            xfer = self.pk_transfer(self.ks_xfer, self.zs_xfer, varied_param_name=varied_param_name, varied_param_value=varied_param_value)
        else:
            xfer = np.ones(len(self.ells))
        cl = A_ksz * (self.ells / 3000)**n_ksz * self.cl_ksz.copy() * xfer
        cl[:2] = 0
        return cl


    def calculate_theory_spectra(self, varied_param_name=None, varied_param_value=None):
        if varied_param_name is not None:
            step_dir = 'up' if (varied_param_value > self.fid_params[varied_param_name]) else 'down'
            print(f'[rank {mpi.rank}] varying {varied_param_name} {step_dir}')
        else:
            step_dir = None
            print(f'[rank {mpi.rank}] calculating fiducial theory')
        # get the kSZ power spectrum
        if self.ksz:
            cl_ksz = self.calculate_cl_ksz(varied_param_name=varied_param_name, varied_param_value=varied_param_value)
        # get the CAMB params instance and results
        pars = self.set_camb_params(varied_param_name=varied_param_name, varied_param_value=varied_param_value)
        results = camb.get_results(pars)
        # calculate clkk
        if self.wdm:
            if varied_param_name == 'm_wdm':
                m_wdm = varied_param_value
            else:
                m_wdm = self.fid_params['m_wdm']
        else:
            m_wdm = None
        if self.xfer:
            pk_transfer_function = lambda k, z: self.pk_transfer(k, z, varied_param_name=varied_param_name, varied_param_value=varied_param_value)
        else:
            pk_transfer_function = None
        clkk = self.calculate_clkk(pars, m_wdm=m_wdm, pk_transfer_function=pk_transfer_function)
        # calculate and save the CMB spectra
        theo = {}
        for cmb_type in self.cmb_types:
            theo[cmb_type] = {'ells': np.arange(self.lmax+1), 'kk': clkk[:self.lmax+1].copy()}
            if cmb_type == 'delensed': # get residual lensing
                cl = theory.get_residual_lensing(clkk.copy(), self.nlkk.copy(), self.Lmin, self.Lmax, len(clkk)-1)
            elif cmb_type == 'unlensed': # don't do any lensing
                cl = np.zeros(clkk.shape)
            else:
                cl = clkk.copy()
            theo_spectra = results.get_lensed_cls_with_spectrum(cl * 4 / (2 * np.pi), lmax=self.lmax, CMB_unit='muK', raw_cl=True)
            for i, s in enumerate(['tt', 'ee', 'bb', 'te']):
                theo[cmb_type][s] = theo_spectra[:,i].copy()
                if self.ksz and (s == 'tt'):
                    theo[cmb_type][s] += cl_ksz
            # save it
            fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, varied_param_name, step_dir, use_H0=False)
            header_info = f'{varied_param_name} = {varied_param_value}\n'
            utils.save_to_file(fname, theo[cmb_type], keys=self.theo_cols, extra_header_info=header_info)
        return theo


    def calculate_clkk(self, camb_params, m_wdm=None, pk_transfer_function=None):
        """
        pk_transfer_function should be a function of arrays k, z; evaluates at pairs (k, z)
        """
        if self.wdm and (m_wdm is None):
            m_wdm = self.fid_params['m_wdm']
        PK = camb.get_matter_power_interpolator(camb_params, nonlinear=True,
                                                hubble_units=False, k_hunit=False,
                                                kmax=1000, zmax=1088,
                                                var1=camb.model.Transfer_Weyl,
                                                var2=camb.model.Transfer_Weyl)
        bg_results = camb.get_background(camb_params)
        # integrate P_psi back to recombination to get clkk:
        chistar = bg_results.conformal_time(0) - bg_results.tau_maxvis
        chis = np.linspace(0, chistar, self.npts_int)
        zs = bg_results.redshift_at_comoving_radial_distance(chis)
        dchis = (chis[2:]-chis[:-2])/2
        chis = chis[1:-1]
        zs = zs[1:-1]
        # do the integral:
        win = ((chistar - chis) / (chis**2 * chistar))**2
        w = np.ones(chis.shape) # this is just used to set to zero k values out of range of interpolation
        ls = np.arange(2, camb_params.max_l+1, dtype=np.float64)
        cl = np.zeros(camb_params.max_l-1)
        for i, l in enumerate(ls):
            k=(l+0.5)/chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=1e3] = 0
            integrand = w * PK.P(zs, k, grid=False) * win / k**4
            if self.wdm:
                integrand *= self.wdm_transfer(k, zs, m_wdm)
            elif pk_transfer_function is not None:
                integrand *= pk_transfer_function(k, zs)
            cl[i] = np.dot(dchis, integrand)
        cl *= (ls*(ls+1))**2
        # return array starting from L = 0:
        clkk = np.zeros(camb_params.max_l+1)
        clkk[2:] = cl
        return clkk


    def save_fisher_derivs(self):
        for param in self.step_sizes.keys():
            delta_param = 2 * self.step_sizes[param]
            header_info = f'{param}: fiducial = {self.fid_params[param]}, step size = {self.step_sizes[param]}\n'
            for cmb_type in self.cmb_types:
                theo_up_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, 'up',  use_H0=False)
                theo_up = utils.load_from_file(theo_up_fname, self.theo_cols)
                theo_down_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, 'down', use_H0=False)
                theo_down = utils.load_from_file(theo_down_fname, self.theo_cols)
                derivs = {'ells': self.ells.copy()}
                for s in self.spectra:
                    derivs[s] = (theo_up[s] - theo_down[s]) / delta_param
                fname = config.fisher_cmb_deriv_fname(self.derivs_dir, cmb_type, param, use_H0=False)
                utils.save_to_file(fname, derivs, keys=self.theo_cols, extra_header_info=header_info)


    def fisher_derivs(self):
        for n, idx in enumerate(self.task_idxs):
            param, value = self.param_values[idx]
            self.calculate_theory_spectra(varied_param_name=param, varied_param_value=value)
        mpi.comm.barrier()
        if mpi.rank == 0:
            self.save_fisher_derivs()
        mpi.comm.barrier()

    def calc_fisher_matrix(self, params=None,
                       priors=None,
                       desi_bao=False,
                       spectra=None):
        if spectra is None:
            spectra = self.spectra
        fmats, fisher_params = calc_fisher_matrix(self.fisher_dir, cmb_types=self.cmb_types, spectra=spectra, params=params, priors=priors, desi_bao=desi_bao, hd_data_version=self.hd_data_version)
        return fmats, fisher_params

    def get_fisher_errors(self, params=None,
                       priors=None,
                       desi_bao=False,
                       cmb_types=None, spectra=None):
        fmats, fisher_params = self.calc_fisher_matrix(params=params, priors=priors, desi_bao=desi_bao, spectra=spectra)
        errors = {}
        for cmb_type in self.cmb_types:
            errors[cmb_type] = fisher.get_fisher_errors(fmats[cmb_type], fisher_params[cmb_type])
        return errors
