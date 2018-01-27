#!/usr/bin/env python
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy import stats
import JAM_mod1.pyjam as pyjam
import JAM.utils.util_dm as util_dm
import JAM.utils.util_mge as util_mge
import JAM.utils.util_gas as util_gas
import JAM.utils.velocity_plot as velocity_plot
from JAM_mod1.utils.util_rst import estimatePrameters, printParameters
from JAM_mod1.utils.util_rst import printModelInfo, printBoundaryPrior
from astropy.cosmology import Planck13
from time import time, localtime, strftime
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
# from emcee.utils import MPIPool
import sys

# parameter boundaries. [lower, upper]
boundary = {'cosinc': [0.0, 1.0], 
            'beta0': [-1.6, 0.4], 'Ra': [1e3, 15e3], 'a': [0.0, 2.0], 
            'logrho_s': [3.0, 10.0], 'rs': [5.0, 45.0], 'gamma': [-1.6, 0.0], 'ml': [0.5, 15],
            'logdelta': [-1.0, 1.0], 'q': [0.1, 0.999], 'logml': [-1.0, 1.15],
            }
# parameter gaussian priors. [mean, sigma]
prior = {'cosinc': [0.0, 1e4], 
         'beta0': [0.0, 1e4], 'Ra': [5e3, 20e4], 'a': [1.0, 1e4], 
         'logrho_s': [5.0, 1e4], 'rs': [10.0, 1e4], 'gamma': [-1.0, 1e4], 'ml': [1.0, 1e4],
         'logdelta': [-1.0, 1e4], 'q': [0.9, 1e4], 'logml': [0.5, 1e4]}

model = {'boundary': boundary, 'prior': prior}


def check_boundary(parsDic, boundary=None):
    '''
    Check whether parameters are within the boundary limits
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      -np.inf or 0.0
    '''
    for key in parsDic.keys():
        if boundary[key][0] < parsDic[key] < boundary[key][1]:
            pass
        else:
            return -np.inf
    return 0.0


def lnprior(parsDic, prior=None):
    '''
    Calculate the gaussian prior lnprob
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      lnprob
    '''
    rst = 0.0
    for key in parsDic.keys():
        rst += -0.5 * (parsDic[key] - prior[key][0])**2/prior[key][1]**2
    return rst


def flat_initp(keys, nwalkers):
    '''
    create initital positions for mcmc. Flat distribution within prior.
    keys: List of parameter name
    nwalkers: number of emcee walkers
    '''
    ndim = len(keys)
    p0 = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        p0[:, i] = np.random.uniform(low=boundary[keys[i]][0]+1e-4,
                                     high=boundary[keys[i]][1]-1e-4,
                                     size=nwalkers)
    return p0


def analyzeRst(sampler, model, nburnin=0):
    '''
    analyze the mcmc and generage resutls
      chain: (nwalker, nstep, ndim)
      lnprobability: (nwalker, nstep)
    '''
    rst = {}
    rst['chain'] = sampler.chain
    rst['lnprobability'] = sampler.lnprobability
    try:
        # rst['acor'] = sampler.acor
        rst['acor'] = sampler.get_autocorr_time(c = 2)
    except:
        rst['acor'] = np.nan
    print('Mean autocorrelation time: {:.1f}'.format(np.mean(rst['acor'])))
    rst['acceptance_fraction'] = sampler.acceptance_fraction
    rst['goodchains'] = ((rst['acceptance_fraction'] > 0.15) *
                         (rst['acceptance_fraction'] < 0.75))
    print('Mean accept fraction: {:.3f}'
          .format(np.mean(rst['acceptance_fraction'])))
    if rst['goodchains'].sum() / float(model['nwalkers']) < 0.6:
        print('Warning - goodchain fraction less than 0.6')
        goodchains = np.ones_like(rst['goodchains'], dtype=bool)
    else:
        goodchains = rst['goodchains']
    flatchain = \
        rst['chain'][goodchains, nburnin:, :].reshape((-1, model['ndim']))
    flatlnprob = rst['lnprobability'][goodchains, nburnin:].reshape(-1)
    medianPars = estimatePrameters(flatchain, flatlnprob=flatlnprob)
    meanPars = estimatePrameters(flatchain, flatlnprob=flatlnprob,
                                 method='mean')
    peakPars = estimatePrameters(flatchain, flatlnprob=flatlnprob,
                                 method='peak')
    maxPars = estimatePrameters(flatchain, flatlnprob=flatlnprob, method='max')
    rst['medianPars'] = medianPars
    rst['meanPars'] = meanPars
    rst['peakPars'] = peakPars
    rst['maxPars'] = maxPars
    print('medianPars')
    printParameters(model['JAMpars'], medianPars)
    print('meanPars')
    printParameters(model['JAMpars'], meanPars)
    print('peakPars')
    printParameters(model['JAMpars'], peakPars)
    print('maxPars')
    printParameters(model['JAMpars'], maxPars)
    return rst


def dump(model):
    with open('{}/{}'.format(model['outfolder'], model['fname']), 'wb') as f:
        pickle.dump(model, f)


def _sigmaClip(sampler, pos):
    maxN = model['clipMaxN']
    lnprob = model['lnprob']
    N = 0
    while True:
        oldGoodbins = model['goodbins'].copy()
        startTime = time()
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, model['clipStep'])
        flatchain = sampler.flatchain
        flatlnprob = sampler.flatlnprobability
        pars = estimatePrameters(flatchain, method='max',
                                 flatlnprob=flatlnprob)
        N += 1
        rmsModel = lnprob(pars, model=model, returnType='rmsModel')
        chi2 = lnprob(pars, model=model, returnType='chi2')
        inThreeSigma = (abs(rmsModel - model['rms']) < model['errRms'] *
                        model['clipSigma'])
        model['goodbins'] *= inThreeSigma
        chi2dof = ((rmsModel[oldGoodbins] - model['rms'][oldGoodbins])**2 /
                   model['errRms'][oldGoodbins]).sum() / oldGoodbins.sum()

        print('--------------------------------------------------')
        print('Time for clip {}: {:.2f}s'.format(N, time()-startTime))
        print('best parameters:')
        printParameters(model['JAMpars'], pars)
        print('Number of old goodbins: {}'.format(oldGoodbins.sum()))
        print('Number of new goodbins: {}'.format(model['goodbins'].sum()))
        print('Chi2: {:.2f}'.format(chi2))
        print('Chi2/dof: {:.3f}'.format(chi2dof))
        if N >= maxN:
            print('Warning - clip more than {}'.format(maxN))
            break
        if np.array_equal(model['goodbins'], oldGoodbins):
            print('Clip srccess')
            break
        if model['goodbins'].sum() / float(model['initGoodbins'].sum()) <\
           model['minFraction']:
            print('clip too many pixels: goodbin fraction < {:.2f}'
                  .format(model['minFraction']))
            break
        sys.stdout.flush()
    sys.stdout.flush()
    return sampler, pos


def _runEmcee(sampler, p0):
    burninStep = model['burnin']
    runStep = model['runStep']
    # burnin
    startTime = time()
    pos, prob, state = sampler.run_mcmc(p0, burninStep)
    print('Start running')
    print('Time for burnin: {:.2f}s'.format(time()-startTime))
    sys.stdout.flush()
    flatchain = sampler.flatchain
    flatlnprob = sampler.flatlnprobability
    pars = estimatePrameters(flatchain, method='max', flatlnprob=flatlnprob)
    printParameters(model['JAMpars'], pars)
    sampler.reset()
    sys.stdout.flush()
    # clip run if set
    if model['clip'] == 'noclip':
        pass
    elif model['clip'] == 'sigma':
        sampler, pos = _sigmaClip(sampler, pos)
    else:
        raise ValueError('clip value {} not supported'
                         .format(model['clip']))
    # Final run
    sampler.run_mcmc(pos, runStep)
    return sampler


def lnprob_massFollowLight(pars, returnType='lnprob', model=None):
    cosinc, beta0, Ra, a, ml = pars
    # print(pars)
    parsDic = {'cosinc': cosinc, 
               'beta0': beta0, 'Ra': Ra, 'a': a, 
               'ml': ml}
    rst = {}
    if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
        rst['lnprob'] = -np.inf
        rst['chi2'] = np.inf
        rst['flux'] = None
        rst['rmsModel'] = None
        rst['dh'] = None
        return rst[returnType]
    lnpriorValue = lnprior(parsDic, prior=model['prior'])
    inc = np.arccos(cosinc)
    JAM = model['JAM']
    rmsModel = JAM.run(inc, beta0, a, Ra, ml=ml)
    chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
             model['errRms'][model['goodbins']])**2).sum()
    if np.isnan(chi2):
        print('Warning - JAM return nan value, beta0={:.2f} may not'
              ' be correct'.format(beta0))
        rst['lnprob'] = -np.inf
        rst['chi2'] = np.inf
        rst['flux'] = None
        rst['rmsModel'] = None
        rst['dh'] = None
        return rst[returnType]

    rst['lnprob'] = -0.5*chi2 + lnpriorValue
    rst['chi2'] = chi2
    rst['flux'] = JAM.flux
    rst['rmsModel'] = rmsModel
    rst['dh'] = None
    return rst[returnType]


def lnprob_spherical_gNFW(pars, returnType='lnprob', model=None):
    cosinc, beta0, Ra, a, ml, logrho_s, rs, gamma = pars # remove a if you do not need it
    # print(pars)
    parsDic = {'cosinc': cosinc, 
               'beta0': beta0, 'Ra': Ra, 'a': a, 
               'ml': ml, 'logrho_s': logrho_s, 'rs': rs, 'gamma': gamma}
    rst = {}
    if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
        rst['lnprob'] = -np.inf
        rst['chi2'] = np.inf
        rst['flux'] = None
        rst['rmsModel'] = None
        rst['dh'] = None
        return rst[returnType]
    lnpriorValue = lnprior(parsDic, prior=model['prior'])
    inc = np.arccos(cosinc)
    JAM = model['JAM']
    dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
    dh_mge3d = dh.mge3d()
    rmsModel = JAM.run(inc, beta0, a, Ra, ml=ml, mge_dh=dh_mge3d)
    chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
             model['errRms'][model['goodbins']])**2).sum()
    if np.isnan(chi2):
        print('Warning - JAM return nan value, beta0={:.2f} may not'
              ' be correct'.format(beta0))
        rst['lnprob'] = -np.inf
        rst['chi2'] = np.inf
        rst['flux'] = None
        rst['rmsModel'] = None
        rst['dh'] = None
        return rst[returnType]

    rst['lnprob'] = -0.5*chi2 + lnpriorValue
    rst['chi2'] = chi2
    rst['flux'] = JAM.flux
    rst['rmsModel'] = rmsModel
    rst['dh'] = dh
    return rst[returnType]


# def lnprob_spherical_gNFW_logml(pars, returnType='lnprob', model=None):
#     cosinc, beta, logml, logrho_s, rs, gamma = pars
#     ml = 10**logml
#     # print(pars)
#     parsDic = {'cosinc': cosinc, 'beta': beta, 'logml': logml,
#                'logrho_s': logrho_s, 'rs': rs, 'gamma': gamma}
#     rst = {}
#     if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]
#     lnpriorValue = lnprior(parsDic, prior=model['prior'])
#     inc = np.arccos(cosinc)
#     Beta = np.zeros(model['lum2d'].shape[0]) + beta
#     JAM = model['JAM']
#     dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
#     dh_mge3d = dh.mge3d()
#     rmsModel = JAM.run(inc, Beta, ml=ml, mge_dh=dh_mge3d)
#     chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
#              model['errRms'][model['goodbins']])**2).sum()
#     if np.isnan(chi2):
#         print('Warning - JAM return nan value, beta={:.2f} may not'
#               ' be correct'.format(beta))
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]

#     rst['lnprob'] = -0.5*chi2 + lnpriorValue
#     rst['chi2'] = chi2
#     rst['flux'] = JAM.flux
#     rst['rmsModel'] = rmsModel
#     rst['dh'] = dh
#     return rst[returnType]


# def lnprob_spherical_gNFW_gradient(pars, returnType='lnprob', model=None):
#     cosinc, beta, ml, logdelta, logrho_s, rs, gamma = pars
#     # print(pars)
#     parsDic = {'cosinc': cosinc, 'beta': beta, 'ml': ml, 'logdelta': logdelta,
#                'logrho_s': logrho_s, 'rs': rs, 'gamma': gamma}
#     rst = {}
#     if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]
#     lnpriorValue = lnprior(parsDic, prior=model['prior'])
#     inc = np.arccos(cosinc)
#     Beta = np.zeros(model['lum2d'].shape[0]) + beta
#     sigma = model['pot2d'][:, 1] / model['Re_arcsec']
#     ML = util_mge.ml_gradient_gaussian(sigma, 10**logdelta, ml0=ml)
#     JAM = model['JAM']
#     dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
#     dh_mge3d = dh.mge3d()
#     rmsModel = JAM.run(inc, Beta, ml=ML, mge_dh=dh_mge3d)
#     chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
#              model['errRms'][model['goodbins']])**2).sum()
#     if np.isnan(chi2):
#         print('Warning - JAM return nan value, beta={:.2f} may not'
#               ' be correct'.format(beta))
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]

#     rst['lnprob'] = -0.5*chi2 + lnpriorValue
#     rst['chi2'] = chi2
#     rst['flux'] = JAM.flux
#     rst['rmsModel'] = rmsModel
#     rst['dh'] = dh
#     return rst[returnType]


# def lnprob_spherical_gNFW_gas(pars, returnType='lnprob', model=None):
#     cosinc, beta, ml, logrho_s, rs, gamma = pars
#     # print(pars)
#     parsDic = {'cosinc': cosinc, 'beta': beta, 'ml': ml,
#                'logrho_s': logrho_s, 'rs': rs, 'gamma': gamma}
#     rst = {}
#     if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]
#     lnpriorValue = lnprior(parsDic, prior=model['prior'])
#     inc = np.arccos(cosinc)
#     Beta = np.zeros(model['lum2d'].shape[0]) + beta
#     JAM = model['JAM']
#     dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
#     dh_mge3d = dh.mge3d()
#     dh_mge3d = np.append(model['gas3d'], dh_mge3d, axis=0)  # add gas mge
#     rmsModel = JAM.run(inc, Beta, ml=ml, mge_dh=dh_mge3d)
#     chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
#              model['errRms'][model['goodbins']])**2).sum()
#     if np.isnan(chi2):
#         print('Warning - JAM return nan value, beta={:.2f} may not'
#               ' be correct'.format(beta))
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]

#     rst['lnprob'] = -0.5*chi2 + lnpriorValue
#     rst['chi2'] = chi2
#     rst['flux'] = JAM.flux
#     rst['rmsModel'] = rmsModel
#     rst['dh'] = dh
#     return rst[returnType]


# def lnprob_spherical_total_dpl(pars, returnType='lnprob', model=None):
#     cosinc, beta, logrho_s, rs, gamma = pars
#     # print(pars)
#     parsDic = {'cosinc': cosinc, 'beta': beta,
#                'logrho_s': logrho_s, 'rs': rs, 'gamma': gamma}
#     rst = {}
#     if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]
#     lnpriorValue = lnprior(parsDic, prior=model['prior'])
#     inc = np.arccos(cosinc)
#     Beta = np.zeros(model['lum2d'].shape[0]) + beta
#     JAM = model['JAM']
#     dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
#     dh_mge3d = dh.mge3d()
#     rmsModel = JAM.run(inc, Beta, ml=0.0, mge_dh=dh_mge3d)
#     chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
#              model['errRms'][model['goodbins']])**2).sum()
#     if np.isnan(chi2):
#         print('Warning - JAM return nan value, beta={:.2f} may not'
#               ' be correct'.format(beta))
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]

#     rst['lnprob'] = -0.5*chi2 + lnpriorValue
#     rst['chi2'] = chi2
#     rst['flux'] = JAM.flux
#     rst['rmsModel'] = rmsModel
#     rst['dh'] = dh
#     return rst[returnType]


# def lnprob_oblate_total_dpl(pars, returnType='lnprob', model=None):
#     cosinc, beta, logrho_s, rs, gamma, q = pars
#     # print(pars)
#     parsDic = {'cosinc': cosinc, 'beta': beta, 'logrho_s': logrho_s,
#                'rs': rs, 'gamma': gamma, 'q': q}
#     rst = {}
#     if np.isinf(check_boundary(parsDic, boundary=model['boundary'])):
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]
#     lnpriorValue = lnprior(parsDic, prior=model['prior'])
#     inc = np.arccos(cosinc)
#     Beta = np.zeros(model['lum2d'].shape[0]) + beta
#     JAM = model['JAM']
#     dh = util_dm.gnfw2d(10**logrho_s, rs, gamma, q)
#     dh_mge3d = dh.mge3d()
#     rmsModel = JAM.run(inc, Beta, ml=0.0, mge_dh=dh_mge3d)
#     chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
#              model['errRms'][model['goodbins']])**2).sum()
#     if np.isnan(chi2):
#         print('Warning - JAM return nan value, beta={:.2f} may not'
#               ' be correct'.format(beta))
#         rst['lnprob'] = -np.inf
#         rst['chi2'] = np.inf
#         rst['flux'] = None
#         rst['rmsModel'] = None
#         rst['dh'] = None
#         return rst[returnType]

#     rst['lnprob'] = -0.5*chi2 + lnpriorValue
#     rst['chi2'] = chi2
#     rst['flux'] = JAM.flux
#     rst['rmsModel'] = rmsModel
#     rst['dh'] = dh
#     return rst[returnType]


class mcmc:
    '''
    input parameter
      galaxy: A python dictionary which contains all the necessary data for
        running JAM model and arguments for running 'emcee'. See the text
        below for a detialed description of the keys. Parameters with * must
        be provided.

    1. Obervitional data
      *lum2d: 2D mge coefficents for galaxy's surface brightness. N*3 array,
        density [L_solar/pc^2], sigma [arcsec], qobs [none]
      pot2d: 2D mge coefficents for galaxy's luminous matter potential. If
        not set, the same value will be used as lum2d.
      distance: Galaxy's distance [Mpc]
      redshift: Galaxy's redshift. (One only need to provied distance or
        redshift for a successful run.
      *xbin: x position of the data points [arcsec], lenght N array.
      *ybin: y position of the data points [arcsec], lenght N array.
      vel: velocity at (x, y) [km/s], lenght N array.
      errVel: velocity error [km/s], lenght N array.
      disp: velocity dispersion at (x, y) [km/s], lenght N array.
      errDisp: velocity dispersion error [km/s], lenght N array.
      *rms: root-mean-squared velocity [km/s], lenght N array.
      errRms: root-mean-squared velocity error [km/s], if rms and errRms
        are not provided, they will be calculated from vel, disp, errVel
        and errDisp, otherwise rms and errRms will be directly used for
        JAM modelling.
      errSacle: errRms *= errSacle. Default: 1.0
      goodbins: bool array, true for goodbins which are used in MCMC
        fitting, lenght N array. Default: all bins are good.
      bh: black hole mass [M_solar], scalar. Default: None
    2. JAM arguments
      sigmapsf: psf for observational data [arcsec], scalar. Default: 0.0
      pixsize: instrument pixsize with which kinematic data are obseved
        [arcsec], scalar. Default: 0.0
      shape: deprojection shape, bool. Default: oblate
      nrad: interpolation grid size, integer. Default: 25
    3. Emcee arguments
      burnin: Number of steps for burnin, integer. Default: 500 steps
      clip: clip run method, string. noclip or sigma. Default: noclip
      clipStep: Number of steps for each clip, integer. Default: 1000 steps
      runStep: Number of steps for the final run, integer. Default: 1000 steps
      nwalkers: Number of walkers in mcmc, integer. Default: 30
      p0: inital position distribution of the mcmc chains. String, flat or fit
    4. Output arguments
      outfolder: output folder path. Default: '.'
      fname: output filename. Default: 'dump.dat'
    '''

    def __init__(self, galaxy):
        '''
        initialize model parameters and data
        '''
        self.lum2d = galaxy['lum2d']
        self.Re_arcsec = util_mge._Re(self.lum2d)
        self.pot2d = galaxy.get('pot2d', self.lum2d.copy())
        self.distance = galaxy.get('distance', None)
        self.redshift = galaxy.get('redshift', None)
        if self.distance is None:
            if self.redshift is None:
                raise RuntimeError('redshift or distance must be provided!')
            else:
                self.distance = \
                    Planck13.angular_diameter_distance(self.redshift).value
        self.xbin = galaxy.get('xbin', None)
        self.ybin = galaxy.get('ybin', None)
        self.vel = galaxy.get('vel', None)
        self.errVel = galaxy.get('errVel', None)
        self.disp = galaxy.get('disp', None)
        self.errDisp = galaxy.get('errDisp', None)
        self.errScale = galaxy.get('errScale', 1.0)
        self.rms = galaxy.get('rms', None)
        if self.rms is None:
            if (self.vel is None) or (self.disp is None):
                raise RuntimeError('rms or (vel, disp) must be provided')
            else:
                self.rms = np.sqrt(self.vel**2 + self.disp**2)
        self.errRms = galaxy.get('errRms', None)
        if self.errRms is None:
            if (self.errVel is not None) and (self.errDisp is not None) and \
                    (self.vel is not None) and (self.disp is not None):
                self.errRms = (np.sqrt((self.errVel*self.vel)**2 +
                                       (self.errDisp*self.disp)**2) /
                               np.sqrt(self.vel**2 + self.disp**2))
            else:
                self.errRms = self.rms*0.0 + np.median(self.rms) * 0.1
        self.errRms *= self.errScale
        self.goodbins = galaxy.get('goodbins',
                                   np.ones_like(self.rms, dtype=bool))
        self.bh = galaxy.get('bh', None)
        self.shape = galaxy.get('shape', 'oblate')

        # save all the model parameters into global dictionary
        global model
        date = strftime('%Y-%m-%d %X', localtime())
        model['date'] = date
        model['name'] = galaxy.get('name', 'LHY')
        model['lum2d'] = self.lum2d
        model['Re_arcsec'] = self.Re_arcsec
        model['pot2d'] = self.pot2d
        model['distance'] = self.distance
        model['redshift'] = self.redshift
        model['xbin'] = self.xbin
        model['ybin'] = self.ybin
        model['vel'] = self.vel
        model['errVel'] = self.errVel
        model['disp'] = self.disp
        model['errDisp'] = self.errDisp
        model['rms'] = self.rms
        model['errRms'] = self.errRms
        model['errScale'] = self.errScale
        model['goodbins'] = self.goodbins
        model['initGoodbins'] = self.goodbins.copy()
        model['bh'] = self.bh
        model['Mgas'] = galaxy.get('Mgas', None)
        model['sigmapsf'] = galaxy.get('sigmapsf', 0.0)
        model['pixsize'] = galaxy.get('pixsize', 0.0)
        model['shape'] = self.shape
        model['nrad'] = galaxy.get('nrad', 25)
        model['burnin'] = galaxy.get('burnin', 500)
        model['clip'] = galaxy.get('clip', 'noclip')
        model['clipStep'] = galaxy.get('clipStep', 1000)
        model['clipSigma'] = galaxy.get('clipSigma', 3.0)
        model['clipMaxN'] = galaxy.get('clipMaxN', 4)
        model['minFraction'] = galaxy.get('minFraction', 0.7)
        model['runStep'] = galaxy.get('runStep', 1000)
        model['nwalkers'] = galaxy.get('nwalkers', 30)
        model['p0'] = galaxy.get('p0', 'flat')
        model['threads'] = galaxy.get('threads', 1)
        model['outfolder'] = galaxy.get('outfolder', './')
        model['fname'] = galaxy.get('fname', 'dump.dat')
        # self.model[''] = self.
        # set cosinc and beta0 priors to aviod JAM crashing
        if self.shape == 'oblate':
            qall = np.append(self.lum2d[:, 2], self.pot2d[:, 2])
            boundary['cosinc'][1] = np.min((qall**2 - 0.003)**0.5)
        elif self.shape == 'prolate':
            qall = np.append(self.lum2d[:, 2], self.pot2d[:, 2])
            if np.any(qall) < 0.101:
                raise ValueError('Input qobs smaller than 0.101 for'
                                 ' prolate model')
            boundary['cosinc'][1] = np.min(((100.0 - 1.0/qall**2)/99.0)**0.5)
            boundary['beta0'][0] = -1.6
            boundary['beta0'][1] = 0.0
        self.startTime = time()
        print('**************************************************')
        print('Initialize mcmc success!')

    def set_boundary(self, key, value):
        '''
        Reset the parameter boundary value
        key: parameter name. Sting
        value: boundary values. length two list
        '''
        if key not in boundary.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('Boundary limits must be a length 2 list')
        print('Change {} limits to [{}, {}], defaults are [{}, {}]'
              .format(key, value[0], value[1], boundary[key][0],
                      boundary[key][1]))
        boundary[key] = value

    def set_prior(self, key, value):
        '''
        Reset the parameter prior value
        key: parameter name. Sting
        value: prior values. length two list
        '''
        if key not in prior.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('prior must be a length 2 list')
        print('Change {} prior to [{}, {}], defaults are [{}, {}]'
              .format(key, value[0], value[1], prior[key][0], prior[key][1]))
        prior[key] = value

    def set_config(self, key, value):
        '''
        Reset the configuration parameter
        key: parameter name. Sting
        value: allowed values
        '''
        if key not in model.keys():
            print('key {} does not exist, create a new key'.format(key))
            model[key] = value
        else:
            print('Change parameter {} to {}, original value is {}'
                  .format(key, value, model[key]))
            model[key] = value

    def get_config(self, key):
        '''
        Get the configuration parameter value of parameter key
        '''
        return model.get(key, None)

    def massFollowLight(self):
        print('--------------------------------------------------')
        print('Mass follow light model')
        model['lnprob'] = lnprob_massFollowLight
        model['type'] = 'massFollowLight'
        model['ndim'] = 5
        model['JAMpars'] = ['cosinc', 'beta0', 'Ra', 'a', 'ml']
        # initialize the JAM class and pass to the global parameter
        model['JAM'] = \
            pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
                              model['distance'],
                              model['xbin'], model['ybin'], mbh=model['bh'],
                              quiet=True, sigmapsf=model['sigmapsf'],
                              pixsize=model['pixsize'], nrad=model['nrad'],
                              shape=model['shape'])
        printModelInfo(model)
        printBoundaryPrior(model)
        nwalkers = model['nwalkers']
        threads = model['threads']
        ndim = model['ndim']
        JAMpars = model['JAMpars']
        if model['p0'] == 'flat':
            p0 = flat_initp(JAMpars, nwalkers)
        elif model['p0'] == 'fit':
            raise ValueError('Calculate maximum lnprob positon from '
                             'optimisiztion - not implemented yet')
        else:
            raise ValueError('p0 must be flat or fit, {} is '
                             'not supported'.format(model['p0']))
        # pool = MPIPool()
        # if not pool.is_master():
        #     pool.wait()
        #     sys.exit(0)
        # Initialize sampler
        initSampler = \
            emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
                                  kwargs={'model': model}, threads=threads)
        sys.stdout.flush()
        sampler = _runEmcee(initSampler, p0)
        # pool.close()
        print('--------------------------------------------------')
        print('Finish! Total elapsed time: {:.2f}s'
              .format(time()-self.startTime))
        rst = analyzeRst(sampler, model)
        sys.stdout.flush()
        model['rst'] = rst
        dump(model)


    def spherical_gNFW(self):
        print('--------------------------------------------------')
        print('spherical gNFW model')
        model['lnprob'] = lnprob_spherical_gNFW
        model['type'] = 'spherical_gNFW'
        model['ndim'] = 8
        model['JAMpars'] = ['cosinc', 'beta0', 'Ra', 'a', 'ml', 'logrho_s', 'rs', 'gamma']
        # initialize the JAM class and pass to the global parameter
        model['JAM'] = \
            pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
                              model['distance'],
                              model['xbin'], model['ybin'], mbh=model['bh'],
                              quiet=True, sigmapsf=model['sigmapsf'],
                              pixsize=model['pixsize'], nrad=model['nrad'],
                              shape=model['shape'])

        printModelInfo(model)
        printBoundaryPrior(model)
        nwalkers = model['nwalkers']
        threads = model['threads']
        ndim = model['ndim']
        JAMpars = model['JAMpars']
        if model['p0'] == 'flat':
            p0 = flat_initp(JAMpars, nwalkers)
        elif model['p0'] == 'fit':
            raise ValueError('Calculate maximum lnprob positon from '
                             'optimisiztion - not implemented yet')
        else:
            raise ValueError('p0 must be flat or fit, {} is '
                             'not supported'.format(model['p0']))
        initSampler = \
            emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
                                  kwargs={'model': model}, threads=threads)
        sys.stdout.flush()
        sampler = _runEmcee(initSampler, p0)
        # pool.close()
        print('--------------------------------------------------')
        print('Finish! Total elapsed time: {:.2f}s'
              .format(time()-self.startTime))
        rst = analyzeRst(sampler, model)
        sys.stdout.flush()
        model['rst'] = rst
        dump(model)

    # def spherical_gNFW_logml(self):
    #     print('--------------------------------------------------')
    #     print('spherical gNFW model (logml)')
    #     model['lnprob'] = lnprob_spherical_gNFW_logml
    #     model['type'] = 'spherical_gNFW_logml'
    #     model['ndim'] = 6
    #     model['JAMpars'] = ['cosinc', 'beta', 'logml', 'logrho_s',
    #                         'rs', 'gamma']
    #     # initialize the JAM class and pass to the global parameter
    #     model['JAM'] = \
    #         pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
    #                           model['distance'],
    #                           model['xbin'], model['ybin'], mbh=model['bh'],
    #                           quiet=True, sigmapsf=model['sigmapsf'],
    #                           pixsize=model['pixsize'], nrad=model['nrad'],
    #                           shape=model['shape'])

    #     printModelInfo(model)
    #     printBoundaryPrior(model)
    #     nwalkers = model['nwalkers']
    #     threads = model['threads']
    #     ndim = model['ndim']
    #     JAMpars = model['JAMpars']
    #     if model['p0'] == 'flat':
    #         p0 = flat_initp(JAMpars, nwalkers)
    #     elif model['p0'] == 'fit':
    #         raise ValueError('Calculate maximum lnprob positon from '
    #                          'optimisiztion - not implemented yet')
    #     else:
    #         raise ValueError('p0 must be flat or fit, {} is '
    #                          'not supported'.format(model['p0']))
    #     initSampler = \
    #         emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
    #                               kwargs={'model': model}, threads=threads)
    #     sys.stdout.flush()
    #     sampler = _runEmcee(initSampler, p0)
    #     # pool.close()
    #     print('--------------------------------------------------')
    #     print('Finish! Total elapsed time: {:.2f}s'
    #           .format(time()-self.startTime))
    #     rst = analyzeRst(sampler, model)
    #     sys.stdout.flush()
    #     model['rst'] = rst
    #     dump(model)

    # def spherical_gNFW_gradient(self):
    #     print('--------------------------------------------------')
    #     print('spherical gNFW model with stellar M*/L gradient')
    #     model['lnprob'] = lnprob_spherical_gNFW_gradient
    #     model['type'] = 'spherical_gNFW_gradient'
    #     model['ndim'] = 7
    #     model['JAMpars'] = ['cosinc', 'beta', 'ml', 'logdelta',
    #                         'logrho_s', 'rs', 'gamma']
    #     # initialize the JAM class and pass to the global parameter
    #     model['JAM'] = \
    #         pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
    #                           model['distance'],
    #                           model['xbin'], model['ybin'], mbh=model['bh'],
    #                           quiet=True, sigmapsf=model['sigmapsf'],
    #                           pixsize=model['pixsize'], nrad=model['nrad'],
    #                           shape=model['shape'])

    #     printModelInfo(model)
    #     printBoundaryPrior(model)
    #     nwalkers = model['nwalkers']
    #     threads = model['threads']
    #     ndim = model['ndim']
    #     JAMpars = model['JAMpars']
    #     if model['p0'] == 'flat':
    #         p0 = flat_initp(JAMpars, nwalkers)
    #     elif model['p0'] == 'fit':
    #         raise ValueError('Calculate maximum lnprob positon from '
    #                          'optimisiztion - not implemented yet')
    #     else:
    #         raise ValueError('p0 must be flat or fit, {} is '
    #                          'not supported'.format(model['p0']))
    #     initSampler = \
    #         emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
    #                               kwargs={'model': model}, threads=threads)
    #     sys.stdout.flush()
    #     sampler = _runEmcee(initSampler, p0)
    #     # pool.close()
    #     print('--------------------------------------------------')
    #     print('Finish! Total elapsed time: {:.2f}s'
    #           .format(time()-self.startTime))
    #     rst = analyzeRst(sampler, model)
    #     sys.stdout.flush()
    #     model['rst'] = rst
    #     dump(model)

    # def spherical_gNFW_gas(self):
    #     print('--------------------------------------------------')
    #     print('spherical gNFW + gas model')
    #     # calculate gas mge profile
    #     if model['Mgas'] is None:
    #         raise RuntimeError('Gas mass must be provided')
    #     gas = util_gas.gas_exp(model['Mgas'])
    #     model['gas3d'] = gas.mge3d
    #     model['lnprob'] = lnprob_spherical_gNFW_gas
    #     model['type'] = 'spherical_gNFW_gas'
    #     model['ndim'] = 6
    #     model['JAMpars'] = ['cosinc', 'beta', 'ml', 'logrho_s', 'rs', 'gamma']
    #     # initialize the JAM class and pass to the global parameter
    #     model['JAM'] = \
    #         pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
    #                           model['distance'],
    #                           model['xbin'], model['ybin'], mbh=model['bh'],
    #                           quiet=True, sigmapsf=model['sigmapsf'],
    #                           pixsize=model['pixsize'], nrad=model['nrad'],
    #                           shape=model['shape'])

    #     printModelInfo(model)
    #     print('Gas Mass: {:.4e}'.format(model['Mgas']))
    #     printBoundaryPrior(model)
    #     nwalkers = model['nwalkers']
    #     threads = model['threads']
    #     ndim = model['ndim']
    #     JAMpars = model['JAMpars']
    #     if model['p0'] == 'flat':
    #         p0 = flat_initp(JAMpars, nwalkers)
    #     elif model['p0'] == 'fit':
    #         raise ValueError('Calculate maximum lnprob positon from '
    #                          'optimisiztion - not implemented yet')
    #     else:
    #         raise ValueError('p0 must be flat or fit, {} is '
    #                          'not supported'.format(model['p0']))
    #     initSampler = \
    #         emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
    #                               kwargs={'model': model}, threads=threads)
    #     sys.stdout.flush()
    #     sampler = _runEmcee(initSampler, p0)
    #     # pool.close()
    #     print('--------------------------------------------------')
    #     print('Finish! Total elapsed time: {:.2f}s'
    #           .format(time()-self.startTime))
    #     rst = analyzeRst(sampler, model)
    #     sys.stdout.flush()
    #     model['rst'] = rst
    #     dump(model)

    # def spherical_total_dpl(self):
    #     print('--------------------------------------------------')
    #     print('spherical double power-law total mass model')
    #     model['lnprob'] = lnprob_spherical_total_dpl
    #     model['type'] = 'spherical_total_dpl'
    #     model['ndim'] = 5
    #     model['JAMpars'] = ['cosinc', 'beta', 'logrho_s', 'rs', 'gamma']
    #     # initialize the JAM class and pass to the global parameter
    #     model['JAM'] = \
    #         pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
    #                           model['distance'],
    #                           model['xbin'], model['ybin'], mbh=model['bh'],
    #                           quiet=True, sigmapsf=model['sigmapsf'],
    #                           pixsize=model['pixsize'], nrad=model['nrad'],
    #                           shape=model['shape'])

    #     printModelInfo(model)
    #     printBoundaryPrior(model)
    #     nwalkers = model['nwalkers']
    #     threads = model['threads']
    #     ndim = model['ndim']
    #     JAMpars = model['JAMpars']
    #     if model['p0'] == 'flat':
    #         p0 = flat_initp(JAMpars, nwalkers)
    #     elif model['p0'] == 'fit':
    #         raise ValueError('Calculate maximum lnprob positon from '
    #                          'optimisiztion - not implemented yet')
    #     else:
    #         raise ValueError('p0 must be flat or fit, {} is '
    #                          'not supported'.format(model['p0']))
    #     initSampler = \
    #         emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
    #                               kwargs={'model': model}, threads=threads)
    #     sys.stdout.flush()
    #     sampler = _runEmcee(initSampler, p0)
    #     # pool.close()
    #     print('--------------------------------------------------')
    #     print('Finish! Total elapsed time: {:.2f}s'
    #           .format(time()-self.startTime))
    #     rst = analyzeRst(sampler, model)
    #     sys.stdout.flush()
    #     model['rst'] = rst
    #     dump(model)

    # def oblate_total_dpl(self):
    #     print('--------------------------------------------------')
    #     print('oblate double power-law total mass model')
    #     model['lnprob'] = lnprob_oblate_total_dpl
    #     model['type'] = 'oblate_total_dpl'
    #     model['ndim'] = 6
    #     model['JAMpars'] = ['cosinc', 'beta', 'logrho_s', 'rs', 'gamma', 'q']
    #     # initialize the JAM class and pass to the global parameter
    #     model['JAM'] = \
    #         pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
    #                           model['distance'],
    #                           model['xbin'], model['ybin'], mbh=model['bh'],
    #                           quiet=True, sigmapsf=model['sigmapsf'],
    #                           pixsize=model['pixsize'], nrad=model['nrad'],
    #                           shape=model['shape'])

    #     printModelInfo(model)
    #     printBoundaryPrior(model)
    #     nwalkers = model['nwalkers']
    #     threads = model['threads']
    #     ndim = model['ndim']
    #     JAMpars = model['JAMpars']
    #     if model['p0'] == 'flat':
    #         p0 = flat_initp(JAMpars, nwalkers)
    #     elif model['p0'] == 'fit':
    #         raise ValueError('Calculate maximum lnprob positon from '
    #                          'optimisiztion - not implemented yet')
    #     else:
    #         raise ValueError('p0 must be flat or fit, {} is '
    #                          'not supported'.format(model['p0']))
    #     initSampler = \
    #         emcee.EnsembleSampler(nwalkers, ndim, model['lnprob'],
    #                               kwargs={'model': model}, threads=threads)
    #     sys.stdout.flush()
    #     sampler = _runEmcee(initSampler, p0)
    #     # pool.close()
    #     print('--------------------------------------------------')
    #     print('Finish! Total elapsed time: {:.2f}s'
    #           .format(time()-self.startTime))
    #     rst = analyzeRst(sampler, model)
    #     sys.stdout.flush()
    #     model['rst'] = rst
    #     dump(model)

    def chi2_spherical_gNFW(self, p0=None, options=None, method='Nelder-Mead'):
        print('--------------------------------------------------')
        print('Minimize chi2 for spherical gNFW model')
        model['lnprob'] = lnprob_spherical_gNFW
        model['type'] = 'spherical_gNFW'
        model['ndim'] = 8
        model['JAMpars'] = ['cosinc', 'beta0', 'Ra', 'a', 'ml', 'logrho_s', 'rs', 'gamma']
        # initialize the JAM class and pass to the global parameter
        model['JAM'] = \
            pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
                              model['distance'],
                              model['xbin'], model['ybin'], mbh=model['bh'],
                              quiet=True, sigmapsf=model['sigmapsf'],
                              pixsize=model['pixsize'], nrad=model['nrad'],
                              shape=model['shape'])
        par = [np.cos(np.radians(80.0)), 0.01, 0.6, 7.9 + np.log10(3.8),
               13.0, -0.2]
        if p0 is None:
            p0 = par
        bounds = [boundary[key] for key in model['JAMpars']]
        if options is None:
            options = {}
            # options['ftol'] = 1e-10
        res = minimize(lnprob_spherical_gNFW, p0, args=(False, True),
                       bounds=bounds, method=method, options=options)
        for i in range(len(p0)):
            print('Init: {:10.6f} fit: {:10.6f}'.format(p0[i], res.x[i]))
        chi2 = lnprob_spherical_gNFW(res.x, False, True)
        print('chi2: {:.4f}'.format(chi2))
        chi2 = lnprob_spherical_gNFW(p0, False, True)
        print('chi2: {:.4f}'.format(chi2))
        # printModelInfo(model)
        # printBoundaryPrior(model)
        # nwalkers = model['nwalkers']
        # threads = model['threads']
        # ndim = model['ndim']
        # JAMpars = model['JAMpars']
        print('--------------------------------------------------')

    def run_spherical_gNFW(self, par, plot=False, save=True, path='./',
                           fname='single_rst', vmap='map', markersize=0.5,
                           rDot=0.24):
        print('--------------------------------------------------')
        print('Run spherical gNFW model with given parameters')
        model['lnprob'] = lnprob_spherical_gNFW
        model['type'] = 'spherical_gNFW'
        model['ndim'] = 8
        model['JAMpars'] = ['cosinc', 'beta0', 'Ra', 'a', 'ml', 'logrho_s', 'rs', 'gamma']
        # initialize the JAM class and pass to the global parameter
        model['JAM'] = \
            pyjam.axi_rms.jam(model['lum2d'], model['pot2d'],
                              model['distance'],
                              model['xbin'], model['ybin'], mbh=model['bh'],
                              quiet=True, sigmapsf=model['sigmapsf'],
                              pixsize=model['pixsize'], nrad=model['nrad'],
                              shape=model['shape'])
        rmsModel = lnprob_spherical_gNFW(par, True, False)
        xbin = model['xbin']
        ybin = model['ybin']
        rms = self.rms
        errRms = self.errRms
        goodbins = self.goodbins
        chi2 = np.sum(((rms[goodbins] - rmsModel[goodbins]) /
                       errRms[goodbins])**2)
        chi2_dof = chi2/goodbins.sum()
        for i in range(len(par)):
            print('{}: {:.4f}'.format(model['JAMpars'][i], par[i]))
        print('chi2: {:.4f}'.format(chi2))
        print('chi2/dof: {:.4f}'.format(chi2_dof))
        print('--------------------------------------------------')

        rst = {'xbin': xbin, 'ybin': ybin, 'rms': rms, 'errRms': errRms,
               'goodbins': goodbins, 'rmsModel': rmsModel, 'chi2': chi2,
               'chi2_dof': chi2_dof, 'pars': par}
        if save:
            with open('{}/{}.dat'.format(path, fname), 'wb') as f:
                pickle.dump(rst, f)
        if plot:
            fig = plt.figure(figsize=(18/1.5, 5./1.5))
            axes0a = fig.add_subplot(131)
            axes0b = fig.add_subplot(132)
            axes0c = fig.add_subplot(133)
            fig.subplots_adjust(left=0.05, bottom=0.1, right=0.92,
                                top=0.99, wspace=0.4)
            vmin, vmax = stats.scoreatpercentile(rms[goodbins],
                                                 [0.5, 99.5])
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            velocity_plot(xbin, ybin, rms, ax=axes0b,
                          text='$\mathbf{V_{rms}: Obs}$', size=rDot,
                          norm=norm,  vmap=vmap,
                          markersize=markersize)
            velocity_plot(xbin, ybin, rmsModel,
                          ax=axes0a, text='$\mathbf{V_{rms}: JAM}$',
                          size=rDot, norm=norm, bar=False, vmap=vmap,
                          markersize=markersize)
            residualValue = rmsModel - rms
            vmax = \
                stats.scoreatpercentile(abs(residualValue[goodbins])
                                        .clip(-100, 100.), 99.5)
            norm_residual = colors.Normalize(vmin=-vmax, vmax=vmax)
            velocity_plot(xbin, ybin, residualValue, ax=axes0c,
                          text='$\mathbf{Residual}$', size=rDot,
                          norm=norm_residual, vmap=vmap,
                          markersize=markersize)
            fig.savefig('{}/{}.png'.format(path, fname), dpi=300)
        return rst
