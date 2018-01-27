#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle
from scipy.stats import gaussian_kde
from scipy import stats
from JAM.utils import util_mge
from JAM.utils import util_dm
from JAM.utils import corner_plot
from . import cap_symmetrize_velfield
symmetrize_velfield = cap_symmetrize_velfield.symmetrize_velfield
from JAM.utils.velocity_plot import velocity_plot
from JAM.utils.vprofile import vprofile
import matplotlib.pyplot as plt
from matplotlib import tri, colors
from matplotlib.patches import Circle
from JAM.utils import util_fig
ticks_font = util_fig.ticks_font
text_font = util_fig.text_font
ticks_font1 = util_fig.ticks_font1
label_font = util_fig.label_font


def printParameters(names, values):
    temp = ['{}: {:.2f}  '.format(names[i], values[i])
            for i in range(len(names))]
    print(''.join(temp))


def printModelInfo(model):
    print('--------------------------------------------------')
    print('Model Info')
    print('pyJAM model run at {}'.format(model['date']))
    print('Galaxy name: {}'.format(model.get('name', None)))
    print('Model type: {}'.format(model.get('type', None)))
    print('Number of tracer MGEs: {}'.format(model['lum2d'].shape[0]))
    print('Number of luminous potential MGEs: {}'
          .format(model['pot2d'].shape[0]))
    print('Number of good observational bins: {}/{}'
          .format(model['goodbins'].sum(), len(model['goodbins'])))
    print('Effective radius: {:.2f} arcsec'.format(model['Re_arcsec']))
    print('errScale {:.2f}'.format(model['errScale']))
    print('Model shape: {}'.format(model['shape']))
    print('Sigmapsf: {:.2f}  Pixelsize: {:.2f}'.format(model['sigmapsf'],
                                                       model['pixsize']))
    print('Burning steps: {}'.format(model['burnin']))
    print('Clip:  {}'.format(model['clip']))
    if model['clip'] in ['sigma']:
        print('Clip steps: {}'.format(model['clipStep']))
        if model['clip'] == 'sigma':
            print('Sigma for sigmaclip: {:.2f}'.format(model['clipSigma']))
    print('nwalkers: {}'.format(model['nwalkers']))
    print('Run steps: {}'.format(model['runStep']))
    print('Initial positons of mcmc chains: {}'.format(model['p0']))
    print('--------------------------------------------------')


def printBoundaryPrior(model):
    JAMpars = model.get('JAMpars', ['cosinc', 'beta0', 'Ra', 'a', 
                                    'logrho_s', 'rs', 'gamma', 'ml'])
    prior = model['prior']
    boundary = model['boundary']
    for name in JAMpars:
        print('{:10s} - prior: {:8.3f} {:10.3e}'
              '    - boundary: [{:8.3f}, {:8.3f}]'
              .format(name, prior[name][0], prior[name][1],
                      boundary[name][0], boundary[name][1]))


def load(name, path='.'):
    with open('{}/{}'.format(path, name), 'rb') as f:
        data = pickle.load(f)
    return data


def estimatePrameters(flatchain, method='median', flatlnprob=None):
    '''
    '''
    if method == 'median':
        return np.percentile(flatchain, 50, axis=0)
    elif method == 'mean':
        return np.mean(flatchain, axis=0)
    elif method == 'peak':
        pars = np.zeros(flatchain.shape[1])
        for i in range(len(pars)):
            xmin = flatchain[:, i].min()
            xmax = flatchain[:, i].max()
            kernel = gaussian_kde(flatchain[:, i])
            x = np.linspace(xmin, xmax, 300)
            prob = kernel(x)
            pars[i] = np.mean(x[prob == prob.max()])
        return pars
    elif method == 'max':
        return np.mean(flatchain[flatlnprob == flatlnprob.max(), :], axis=0)
    else:
        raise ValueError('Do not support {} method'.format(method))


class modelRst(object):

    def __init__(self, name, path='.', burnin=0, best='median'):
        self.data = load(name, path=path)
        # load model data into class
        self.ndim = self.data['ndim']
        self.nwalkers = self.data['nwalkers']
        self.chain = self.data['rst']['chain']
        self.goodchains = self.data['rst']['goodchains']
        # check good chain fraction
        if self.goodchains.sum()/float(self.nwalkers) < 0.6:
            self.goodchains = np.ones_like(self.goodchains, dtype=bool)
            print('Warning - goodchain fraction less than 0.6')
            print('Acceptance fraction:')
            print(self.data['rst']['acceptance_fraction'])
        self.lnprob = self.data['rst']['lnprobability']
        self.flatchain = self.chain[self.goodchains,
                                    burnin:, :].reshape(-1, self.ndim)
        self.flatlnprob = self.lnprob[self.goodchains, burnin:].reshape(-1)
        # estimate the beset model parameters
        self.medianPars = estimatePrameters(self.flatchain,
                                            flatlnprob=self.flatlnprob)
        self.meanPars = estimatePrameters(self.flatchain,
                                          flatlnprob=self.flatlnprob,
                                          method='mean')
        self.peakPars = estimatePrameters(self.flatchain,
                                          flatlnprob=self.flatlnprob,
                                          method='peak')
        self.maxPars = estimatePrameters(self.flatchain,
                                         flatlnprob=self.flatlnprob,
                                         method='max')
        # estimate the errors
        percentile = np.percentile(self.flatchain, [16, 50, 84], axis=0)
        self.errPars = np.zeros([2, percentile.shape[1]])
        self.errPars[0, :] = percentile[0, :] - percentile[1, :]
        self.errPars[1, :] = percentile[2, :] - percentile[1, :]
        # choose the best parameter type
        switch = {'median': self.medianPars, 'mean': self.meanPars,
                  'peak': self.peakPars, 'max': self.maxPars}
        bestPars = switch[best]
        self.bestPars_list = bestPars
        self.bestPars = {}
        for i, key in enumerate(self.data['JAMpars']):
            self.bestPars[key] = bestPars[i]
        # model inclination
        cosinc = self.bestPars.get('cosinc', np.pi/2.0)
        self.inc = np.arccos(cosinc)
        if 'ml' in self.bestPars.keys():
            self.ml = self.bestPars['ml']
        else:
            if 'logml' in self.bestPars.keys():
                self.ml = 10**self.bestPars['logml']
            else:
                print('Waring - do not find ml parameter, set to 1.0')
                self.ml = 1.0

        # load observational data
        self.dist = self.data['distance']
        self.pc = self.dist * np.pi / 0.648
        self.lum2d = self.data['lum2d']
        self.pot2d = self.data['pot2d']

        self.xbin = self.data['xbin']
        self.ybin = self.data['ybin']
        self.rms = self.data['rms'].clip(0.0, 600.0)
        self.goodbins = self.data['goodbins']
        self.symetrizedRms = self.rms.copy()
        self.symetrizedRms[self.goodbins] = \
            symmetrize_velfield(self.xbin[self.goodbins],
                                self.ybin[self.goodbins],
                                self.rms[self.goodbins])

        # run a JAM model with the choosen best model parameters
        JAMmodel = self.data['JAM']  # restore JAM model
        JAMlnprob = self.data['lnprob']
        self.rmsModel = JAMlnprob(bestPars, model=self.data,
                                  returnType='rmsModel')
        self.flux = JAMlnprob(bestPars, model=self.data,
                              returnType='flux')
        self.shape = self.data['shape']
        # create stellar mass mge objected (used in mass profile)
        self.LmMge = util_mge.mge(self.pot2d, inc=self.inc, shape=self.shape,
                                  dist=self.dist)
        # set black hole mge object
        bh3dmge = JAMmodel.mge_bh
        if bh3dmge is None:
            self.BhMge = None
        else:
            bh2dmge = util_mge.projection(bh3dmge, inc=self.inc)
            self.BhMge = util_mge.mge(bh2dmge, inc=self.inc)

        if self.data['type'] == 'massFollowLight':
            self.labels = [r'$\mathbf{cosi}$', 
                           r'$\mathbf{\beta0}$', r'$R_a$', r'$a$', 
                           r'$\mathbf{M/L}$']
            self.DmMge = None  # no dark halo
        elif self.data['type'] == 'spherical_gNFW':
            self.labels = [r'$\mathbf{cosi}$', 
                           r'$\mathbf{\beta0}$', r'$R_a$', r'$a$', 
                           r'$\mathbf{M^*/L}$', r'$\mathbf{log\ \rho_s}$',
                           r'$\mathbf{r_s}$', r'$\mathbf{\gamma}$']
            # create dark halo mass mge object
            dh = JAMlnprob(bestPars, model=self.data, returnType='dh')
            dh_mge3d = dh.mge3d()
            self.DmMge = util_mge.mge(dh.mge2d(), inc=self.inc)
        # elif self.data['type'] == 'spherical_gNFW_logml':
        #     self.labels = [r'$\mathbf{cosi}$', r'$\mathbf{\beta}$',
        #                    r'$\mathbf{logM^*/L}$', r'$\mathbf{log\ \rho_s}$',
        #                    r'$\mathbf{r_s}$', r'$\mathbf{\gamma}$']
        #     # create dark halo mass mge object
        #     dh = JAMlnprob(bestPars, model=self.data, returnType='dh')
        #     dh_mge3d = dh.mge3d()
        #     self.DmMge = util_mge.mge(dh.mge2d(), inc=self.inc)
        # elif self.data['type'] == 'spherical_gNFW_gas':
        #     inc = self.inc
        #     Beta = np.zeros(self.lum2d.shape[0]) + bestPars[1]
        #     ml = bestPars[2]
        #     logrho_s = bestPars[3]
        #     rs = bestPars[4]
        #     gamma = bestPars[5]
        #     dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
        #     dh_mge3d = dh.mge3d()
        #     gas3d = self.data['gas3d']
        #     dh_mge3d = np.append(gas3d, dh_mge3d, axis=0)
        #     self.rmsModel = JAMmodel.run(inc, Beta, ml=ml, mge_dh=dh_mge3d)
        #     self.flux = JAMmodel.flux
        #     self.labels = [r'$\mathbf{cosi}$', r'$\mathbf{\beta}$',
        #                    r'$\mathbf{M^*/L}$', r'$\mathbf{log\ \rho_s}$',
        #                    r'$\mathbf{r_s}$', r'$\mathbf{\gamma}$']
        #     # create dark halo mass mge object
        #     self.DmMge = util_mge.mge(dh.mge2d(), inc=self.inc)
        # elif self.data['type'] == 'spherical_gNFW_gradient':
        #     self.labels = [r'$\mathbf{cosi}$', r'$\mathbf{\beta}$',
        #                    r'$\mathbf{M^*/L}$',
        #                    r'$\mathbf{\log \Delta_{IMF}}$',
        #                    r'$\mathbf{log\ \rho_s}$',
        #                    r'$\mathbf{r_s}$', r'$\mathbf{\gamma}$']
        #     # create dark halo mass mge object
        #     dh = JAMlnprob(bestPars, model=self.data, returnType='dh')
        #     dh_mge3d = dh.mge3d()
        #     self.DmMge = util_mge.mge(dh.mge2d(), inc=self.inc)
        # elif self.data['type'] == 'spherical_total_dpl':
        #     self.labels = [r'$\mathbf{cosi}$', r'$\mathbf{\beta}$',
        #                    r'$\mathbf{log\ \rho_s}$',
        #                    r'$\mathbf{r_s}$', r'$\mathbf{\gamma}$']
        #     # create dark halo mass mge object
        #     dh = JAMlnprob(bestPars, model=self.data, returnType='dh')
        #     dh_mge3d = dh.mge3d()
        #     self.DmMge = util_mge.mge(dh.mge2d(), inc=self.inc)
        # elif self.data['type'] == 'oblate_total_dpl':
        #     self.labels = [r'$\mathbf{cosi}$', r'$\mathbf{\beta}$',
        #                    r'$\mathbf{log\ \rho_s}$', r'$\mathbf{r_s}$',
        #                    r'$\mathbf{\gamma}$', r'$\mathbf{q}$']
        #     # create dark halo mass mge object
        #     dh = JAMlnprob(bestPars, model=self.data, returnType='dh')
        #     dh_mge3d = dh.mge3d()
        #     self.DmMge = util_mge.mge(dh.mge2d(self.inc), inc=self.inc)
        else:
            raise ValueError('model type {} not supported'
                             .format(self.data['type']))

    def printInfo(self):
        printModelInfo(self.data)

    def printPrior(self):
        printBoundaryPrior(self.data)

    def meanDisp(self, R):
        '''
        Calcualte the surface brightness weighted mean velocity dispersion
          within R [arcsec]
        '''
        mge = util_mge.mge(self.lum2d, inc=self.inc, shape=self.shape,
                           dist=self.dist)
        surf = mge.surfaceBrightness(self.xbin*self.pc, self.ybin*self.pc)
        r = np.sqrt(self.xbin**2 + self.ybin**2)
        i_in = (r < R) * self.goodbins
        sigma_R = np.average(self.rms[i_in], weights=surf[i_in])
        return sigma_R

    def lambda_R(self, R):
        r = np.sqrt(self.xbin**2 + self.ybin**2)
        vel = self.data['vel']
        i_R = (r < R) * self.goodbins
        if vel is None:
            return np.nan
        else:
            rst = np.average(r[i_R]*abs(vel[i_R]), weights=self.flux[i_R]) /\
                np.average(r[i_R]*self.rms[i_R], weights=self.flux[i_R])
            return rst

    def cornerPlot(self, figname='mcmc.png', outpath='.',
                   clevel=[0.683, 0.95, 0.997], truths='max', true=None,
                   hbins=30, color=[0.8936, 0.5106, 0.2553], vmap='dots',
                   xpos=0.65, ypos=0.58, size=0.2, symetrize=False,
                   residual=True, markersize=1.0, **kwargs):
        switch = {'median': self.medianPars, 'mean': self.meanPars,
                  'peak': self.peakPars, 'max': self.maxPars,
                  'true': true}
        truthsValue = switch[truths]
        kwargs['labels'] = kwargs.get('labels', self.labels)
        fig = corner_plot.corner(self.flatchain, clevel=clevel, hbins=hbins,
                                 truths=truthsValue, color=color,
                                 quantiles=[0.16, 0.5, 0.84], **kwargs)
        # plot velocity map
        if vmap in ['dots', 'map']:
            rDot = kwargs.get('rDot', 0.24)
            axes0a = fig.add_axes([xpos, ypos, size, size])
            axes0b = fig.add_axes([xpos, ypos+size, size, size])
            axes0b.set_yticklabels([])
            axes0b.set_xticklabels([])
            util_fig.set_labels(axes0a)
            if residual:
                axes0c = fig.add_axes([xpos-size*1.3, ypos+size, size, size])
                axes0c.set_yticklabels([])
                axes0c.set_xticklabels([])

            if symetrize:
                rms = self.symetrizedRms
            else:
                rms = self.rms
            vmin, vmax = stats.scoreatpercentile(rms[self.goodbins],
                                                 [0.5, 99.5])
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            # plot mge contours
            dither = 1e-3 * np.random.random(len(self.xbin))
            triangles = tri.Triangulation(self.xbin+dither, self.ybin)
            axes0b.tricontour(triangles,
                              -2.5*np.log10(self.flux/np.max(self.flux)),
                              levels=(np.arange(0, 10)), colors='k')
            axes0a.tricontour(triangles,
                              -2.5*np.log10(self.flux/np.max(self.flux)),
                              levels=(np.arange(0, 10)), colors='k')
            # mark badbins
            for i in range(self.xbin[~self.goodbins].size):
                circle = Circle(xy=(self.xbin[~self.goodbins][i],
                                    self.ybin[~self.goodbins][i]),
                                fc='w', radius=rDot*0.8, zorder=10, lw=0.)
                axes0b.add_artist(circle)

            velocity_plot(self.xbin, self.ybin, rms, ax=axes0b,
                          text='$\mathbf{V_{rms}: Obs}$', size=rDot,
                          norm=norm, bar=False, vmap=vmap,
                          markersize=markersize)
            velocity_plot(self.xbin, self.ybin, self.rmsModel,
                          ax=axes0a, text='$\mathbf{V_{rms}: JAM}$',
                          size=rDot, norm=norm, vmap=vmap,
                          markersize=markersize)
            if residual:
                residualValue = self.rmsModel - rms
                vmax = \
                    stats.scoreatpercentile(abs(residualValue[self.goodbins])
                                            .clip(-100, 100.), 99.5)
                norm_residual = colors.Normalize(vmin=-vmax, vmax=vmax)
                velocity_plot(self.xbin, self.ybin, residualValue, ax=axes0c,
                              text='$\mathbf{Residual}$', size=rDot,
                              norm=norm_residual, vmap=vmap,
                              markersize=markersize)
        else:
            raise ValueError('vmap {} not supported'.format(vmap))
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)

    def plotChain(self, figname='chain.png', outpath='.', **kwargs):
        figsize = (8.0, self.ndim*2.0)
        fig, axes = plt.subplots(self.ndim, 1, sharex=True, figsize=figsize)
        for i in range(self.ndim):
            axes[i].plot(self.chain[:, :, i].T, color='k', alpha=0.2)
            axes[i].set_ylabel(self.labels[i])
        axes[-1].set_xlabel('nstep')
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)

    def plotVrms(self, figname='rmsProfile.png', outpath='.', width=1.0,
                 **kwargs):
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        vprofile(self.xbin, self.ybin, self.rms, vModel=self.rmsModel,
                 ax=ax[0], width=width)
        vprofile(self.xbin, self.ybin, self.rms, vModel=self.rmsModel,
                 ax=ax[1], angle=90.0, width=width, ylabel=None,
                 xlabel='y arcsec')
        plt.tight_layout()
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)

    def dump(self, outpath, name='rst.dat'):
        data = {}
        chi2 = np.sum((((self.data['rms'] - self.rmsModel) /
                        self.data['errRms'])**2)[self.goodbins]) /\
            self.goodbins.sum()
        data['chi2dof'] = chi2
        data['Re_arcsec'] = self.data['Re_arcsec']
        data['dist'] = self.dist
        data['JAMpars'] = self.data['JAMpars']
        data['medianPars'] = self.medianPars
        data['meanPars'] = self.meanPars
        data['peakPars'] = self.peakPars
        data['maxPars'] = self.maxPars
        data['bestPars'] = self.bestPars_list
        data['errPars'] = self.errPars
        data['xbin'] = self.xbin
        data['ybin'] = self.ybin
        data['rms'] = self.data['rms']
        data['errRms'] = self.data['errRms']
        data['rmsModel'] = self.rmsModel
        data['flux'] = self.flux
        data['lum2d'] = self.lum2d
        data['pot2d'] = self.pot2d
        if self.DmMge is None:
            data['dhmge2d'] = None
        else:
            data['dhmge2d'] = self.DmMge.mge2d
        data['Disp_Re'] = self.meanDisp(data['Re_arcsec'])
        data['lambda_Re'] = self.lambda_R(data['Re_arcsec'])
        with open('{}/{}'.format(outpath, name), 'wb') as f:
            pickle.dump(data, f)
        # save some info into txt file
        f = open('{}/rst.txt'.format(outpath), 'w')
        temp = ['{}: {:.3f}_{:+.3f}^{:+.3f}\n'
                .format(data['JAMpars'][i], data['bestPars'][i],
                        data['errPars'][0, i], data['errPars'][1, i])
                for i in range(len(data['JAMpars']))]
        f.write('dist: {:.3f} Mpc\n'.format(data['dist']))
        f.write('Re_arcsec: {:.3f} arcsec\n'.format(data['Re_arcsec']))
        f.write('sigma_Re: {:.3f} km/s\n'.format(data['Disp_Re']))
        f.write('lambda_Re: {:.3f}\n'.format(data['lambda_Re']))
        f.write('chi2/dof: {:.4f}\n'.format(data['chi2dof']))
        f.write('chi2: {:.4f}\n'.format(data['chi2dof']*self.goodbins.sum()))
        f.write(''.join(temp))
        f.close()
