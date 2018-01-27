#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
from JAM.utils import util_mge
from JAM.utils import util_dm
import matplotlib.pyplot as plt
from JAM.utils import util_fig
from JAM_mod1.utils.util_rst import modelRst
from scipy import interpolate
import pickle
ticks_font = util_fig.ticks_font
text_font = util_fig.text_font
ticks_font1 = util_fig.ticks_font1
label_font = util_fig.label_font
ticks_font.set_size(11)
text_font.set_size(14)
label_font.set_size(14)


def _extractProfile(mge, r):
    '''
    mge is the mge object, defined in util_mge.py
    r is the radius in kpc, 1d array
    '''
    mass = mge.enclosed3Dluminosity(r*1e3)
    density = mge.meanDensity(r*1e3) * 1e9
    return mass, density


def _plotProfile(r, profiles, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(r, profiles, **kwargs)


class profile(modelRst):
    def __init__(self, name, path='.', burnin=0, nlines=200, r=None):
        super(profile, self).__init__(name, path=path, burnin=burnin,
                                      best='median')
        self.profiles = {}  # dictionary containing profiles
        if r is None:
            r = np.logspace(np.log10(0.5), np.log10(100.0), 100)
        self.profiles['r'] = r
        # select a subchain to calculate density profile
        ntotal = self.flatchain.shape[0]
        step = ntotal // nlines
        if step == 0:
            print('Warning - nlines > total number of samples')
            step = 1
        ii = np.zeros(ntotal, dtype=bool)
        ii[::step] = True
        self.profiles['nprofiles'] = ii.sum()

        if self.data['type'] in ['spherical_gNFW', 'spherical_gNFW_logml',
                                 'spherical_gNFW_gas']:
            # Calculate stellar mass profiles
            stellarProfiles = np.zeros([len(r), ii.sum(), 2])
            inc = np.arccos(self.flatchain[ii, 0].ravel())
            if self.data['type'] == 'spherical_gNFW_logml':
                mls = 10**self.flatchain[ii, -4].ravel()
            else:
                mls = self.flatchain[ii, -4].ravel()
            mass, density = _extractProfile(self.LmMge, r)
            for i in range(ii.sum()):
                stellarProfiles[:, i, 0] = mls[i] * density
                stellarProfiles[:, i, 1] = mls[i] * mass
            self.profiles['stellar'] = stellarProfiles
            # Calculate dark matter profiles
            # if self.DmMge is None:
            #     raise ValueError('No dark matter halo is found')
            darkProfiles = np.zeros_like(stellarProfiles)
            logrho_s = self.flatchain[ii, -3].ravel()
            rs = self.flatchain[ii, -2].ravel()
            gamma = self.flatchain[ii, -1].ravel()
            for i in range(ii.sum()):
                tem_dh = util_dm.gnfw1d(10**logrho_s[i], rs[i], gamma[i])
                tem_mge = util_mge.mge(tem_dh.mge2d(), inc=inc[i])
                mass, density = _extractProfile(tem_mge, r)
                darkProfiles[:, i, 0] = density
                darkProfiles[:, i, 1] = mass
            self.profiles['dark'] = darkProfiles
            totalProfiles = stellarProfiles + darkProfiles
            self.profiles['total'] = totalProfiles

        # elif self.data['type'] == 'spherical_gNFW_gradient':
        #     # Calculate stellar mass profiles
        #     stellarProfiles = np.zeros([len(r), ii.sum(), 2])
        #     inc = np.arccos(self.flatchain[ii, 0].ravel())
        #     mls = self.flatchain[ii, 2].ravel()
        #     pot_ng = self.pot2d.copy()
        #     pot_tem = np.zeros([1, 3])
        #     sigma = pot_ng[:, 1]/self.data['Re_arcsec']
        #     logdelta = self.flatchain[ii, 3].ravel()
        #     mass = np.zeros([len(r), pot_ng.shape[0]])
        #     density = np.zeros([len(r), pot_ng.shape[0]])
        #     for i in range(pot_ng.shape[0]):
        #         pot_tem[:, :] = pot_ng[i, :]
        #         mge_pot = util_mge.mge(pot_tem, inc=self.inc,
        #                                shape=self.shape, dist=self.dist)
        #         mass[:, i], density[:, i] = _extractProfile(mge_pot, r)
        #     for i in range(ii.sum()):
        #         ML = util_mge.ml_gradient_gaussian(sigma, 10**logdelta[i],
        #                                            ml0=mls[i])
        #         stellarProfiles[:, i, 0] = np.sum(ML * density, axis=1)
        #         stellarProfiles[:, i, 1] = np.sum(ML * mass, axis=1)
        #     self.profiles['stellar'] = stellarProfiles

        #     # Calculate dark matter profiles
        #     darkProfiles = np.zeros_like(stellarProfiles)
        #     logrho_s = self.flatchain[ii, 4].ravel()
        #     rs = self.flatchain[ii, 5].ravel()
        #     gamma = self.flatchain[ii, 6].ravel()
        #     for i in range(ii.sum()):
        #         tem_dh = util_dm.gnfw1d(10**logrho_s[i], rs[i], gamma[i])
        #         tem_mge = util_mge.mge(tem_dh.mge2d(), inc=inc[i])
        #         mass, density = _extractProfile(tem_mge, r)
        #         darkProfiles[:, i, 0] = density
        #         darkProfiles[:, i, 1] = mass
        #     self.profiles['dark'] = darkProfiles
        #     totalProfiles = stellarProfiles + darkProfiles
        #     self.profiles['total'] = totalProfiles

        # elif self.data['type'] == 'spherical_total_dpl':
        #     self.profiles['stellar'] = None
        #     self.profiles['dark'] = None
        #     totalProfiles = np.zeros([len(r), ii.sum(), 2])
        #     inc = np.arccos(self.flatchain[ii, 0].ravel())
        #     logrho_s = self.flatchain[ii, 2].ravel()
        #     rs = self.flatchain[ii, 3].ravel()
        #     gamma = self.flatchain[ii, 4].ravel()
        #     for i in range(ii.sum()):
        #         tem_dh = util_dm.gnfw1d(10**logrho_s[i], rs[i], gamma[i])
        #         tem_mge = util_mge.mge(tem_dh.mge2d(), inc=inc[i])
        #         mass, density = _extractProfile(tem_mge, r)
        #         totalProfiles[:, i, 0] = density
        #         totalProfiles[:, i, 1] = mass
        #     self.profiles['total'] = totalProfiles

        # elif self.data['type'] == 'oblate_total_dpl':
        #     self.profiles['stellar'] = None
        #     self.profiles['dark'] = None
        #     totalProfiles = np.zeros([len(r), ii.sum(), 2])
        #     inc = np.arccos(self.flatchain[ii, 0].ravel())
        #     logrho_s = self.flatchain[ii, 2].ravel()
        #     rs = self.flatchain[ii, 3].ravel()
        #     gamma = self.flatchain[ii, 4].ravel()
        #     q = self.flatchain[ii, 5].ravel()
        #     for i in range(ii.sum()):
        #         tem_dh = util_dm.gnfw2d(10**logrho_s[i], rs[i], gamma[i], q[i])
        #         tem_mge = util_mge.mge(tem_dh.mge2d(inc[i]), inc=inc[i])
        #         mass, density = _extractProfile(tem_mge, r)
        #         totalProfiles[:, i, 0] = density
        #         totalProfiles[:, i, 1] = mass
        #     self.profiles['total'] = totalProfiles
        else:
            raise ValueError('model type {} not supported'
                             .format(self.data['type']))

        # add black hole mass
        if self.data['bh'] is not None:
            self.profiles['total'][:, :, 1] += self.data['bh']

        # add gas component
        self.gas3d = self.data.get('gas3d', None)

        # calculate mass within Re
        Re_kpc = self.data['Re_arcsec'] * self.pc / 1e3
        stellar_Re, dark_Re, total_Re, fdm_Re = self.enclosed3DMass(Re_kpc)
        MassRe = {}
        MassRe['Re_kpc'] = Re_kpc
        MassRe['stellar'] = np.log10(stellar_Re)
        MassRe['dark'] = np.log10(dark_Re)
        MassRe['total'] = np.log10(total_Re)
        MassRe['fdm'] = fdm_Re
        self.profiles['MassRe'] = MassRe

    def save(self, fname='profiles.dat', outpath='.'):
        with open('{}/{}'.format(outpath, fname), 'wb') as f:
            pickle.dump(self.profiles, f)

    def enclosed3DMass(self, r):
        if self.profiles['total'] is not None:
            total_median = np.percentile(self.profiles['total'][:, :, 1],
                                         50, axis=1)
            ftotal = \
                interpolate.interp1d(self.profiles['r'], total_median,
                                     kind='linear', bounds_error=False,
                                     fill_value=np.nan)
            total = ftotal(r)
        else:
            total = np.nan

        if self.profiles['stellar'] is not None:
            stellar_median = np.percentile(self.profiles['stellar'][:, :, 1],
                                           50, axis=1)
            fstellar = \
                interpolate.interp1d(self.profiles['r'], stellar_median,
                                     kind='linear', bounds_error=False,
                                     fill_value=np.nan)
            stellar = fstellar(r)
        else:
            stellar = np.nan

        if self.profiles['dark'] is not None:
            dark_median = np.percentile(self.profiles['dark'][:, :, 1],
                                        50, axis=1)
            fdark = \
                interpolate.interp1d(self.profiles['r'], dark_median,
                                     kind='linear', bounds_error=False,
                                     fill_value=np.nan)
            dark = fdark(r)
        else:
            dark = np.nan
        fdm = dark / total
        return stellar, dark, total, fdm

    def plotProfiles(self, outpath='.', figname='profiles.png', Range=None,
                     true=None, nre=3.5, **kwargs):
        Re_kpc = self.data['Re_arcsec'] * self.pc / 1e3
        dataRange = np.percentile(np.sqrt(self.xbin**2+self.ybin**2), 95) * \
            self.pc / 1e3
        if Range is None:
            Range = [0.5, nre*Re_kpc]
        MassRe = self.profiles['MassRe']
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        fig.subplots_adjust(left=0.12, bottom=0.08, right=0.98,
                            top=0.98, wspace=0.1, hspace=0.1)
        r = self.profiles['r']
        ii = (r > Range[0]) * (r < Range[1])
        logr = np.log10(r[ii])
        # plot stellar density
        if self.profiles['stellar'] is not None:
            stellarDens = np.log10(self.profiles['stellar'][ii, :, 0])
            stellarMass = np.log10(self.profiles['stellar'][ii, :, 1])
            axes[0].plot(logr, stellarDens, 'y', alpha=0.1, **kwargs)
            axes[1].plot(logr, stellarMass, 'y', alpha=0.1, **kwargs)
        # plot dark density
        if self.profiles['dark'] is not None:
            darkDens = np.log10(self.profiles['dark'][ii, :, 0])
            darkMass = np.log10(self.profiles['dark'][ii, :, 1])
            axes[0].plot(logr, darkDens, 'r', alpha=0.1, **kwargs)
            axes[1].plot(logr, darkMass, 'r', alpha=0.1, **kwargs)
        # plot total density
        if self.profiles['total'] is not None:
            totalDens = np.log10(self.profiles['total'][ii, :, 0])
            totalMass = np.log10(self.profiles['total'][ii, :, 1])
            axes[0].plot(logr, totalDens, 'c', alpha=0.1, **kwargs)
            axes[1].plot(logr, totalMass, 'c', alpha=0.1, **kwargs)
        # axes[1].plot(np.log10(Re_kpc), MassRe['total'], 'ok')
        # axes[1].plot(np.log10(Re_kpc), MassRe['stellar'], 'oy')
        # axes[1].plot(np.log10(Re_kpc), MassRe['dark'], 'or')
        util_fig.set_labels(axes[0])
        util_fig.set_labels(axes[1])
        axes[1].set_xlabel(r'$\mathbf{log_{10}\ \ \! R \, \ [kpc]}$',
                           fontproperties=label_font)
        axes[1].set_ylabel(r'$\mathbf{log_{10}\ \ \! M(R) \, \  [M_{\odot}]}$',
                           fontproperties=label_font)
        axes[0].set_ylabel(r'$\mathbf{log_{10}\ \ \! \rho \,'
                           ' \ [M_{\odot}\ \ \! kpc^{-3}]}$',
                           fontproperties=label_font)
        # plot gas density
        if self.gas3d is not None:
            gas2d = util_mge.projection(self.gas3d, self.inc)
            GasMge = util_mge.mge(gas2d, self.inc)
            mass, density = _extractProfile(GasMge, r)
            GasProfile = np.zeros([1, len(r), 2])
            GasProfile[0, :, 0] = density
            GasProfile[0, :, 1] = mass
            self.profiles['gas'] = GasProfile
            axes[0].plot(logr, np.log10(density[ii]), 'b', alpha=0.8, **kwargs)
            axes[1].plot(logr, np.log10(mass[ii]), 'b', alpha=0.8, **kwargs)
        if true is not None:
            rtrue = true.get('r', None)
            if rtrue is None:
                raise RuntimeError('r must be provided for true profile')
            ii = (rtrue > Range[0]) * (rtrue < Range[1])
            if 'stellarDens' in true.keys():
                axes[0].plot(np.log10(rtrue[ii]), true['stellarDens'][ii],
                             'oy', markeredgecolor='k')
            if 'stellarMass' in true.keys():
                axes[1].plot(np.log10(rtrue[ii]), true['stellarMass'][ii],
                             'oy', markeredgecolor='k')
            if 'darkDens' in true.keys():
                axes[0].plot(np.log10(rtrue[ii]), true['darkDens'][ii], 'or',
                             markeredgecolor='k')
            if 'darkMass' in true.keys():
                axes[1].plot(np.log10(rtrue[ii]), true['darkMass'][ii], 'or',
                             markeredgecolor='k')
            if 'totalDens' in true.keys():
                axes[0].plot(np.log10(rtrue[ii]), true['totalDens'][ii], 'oc',
                             markeredgecolor='k')
            if 'totalMass' in true.keys():
                axes[1].plot(np.log10(rtrue[ii]), true['totalMass'][ii], 'oc',
                             markeredgecolor='k')
        for ax in axes:
            ax.set_xlim([np.min(logr), np.max(logr)])
            ax.axvline(np.log10(Re_kpc), ls="dashed", color='b', linewidth=2)
            ax.axvline(np.log10(dataRange), ls="dashed", color='g',
                       linewidth=2)
        axes[0].text(0.05, 0.05, '$\mathbf{M^*(R_e)}$: %4.2f'
                     % (MassRe['stellar']), transform=axes[0].transAxes,
                     fontproperties=text_font)
        axes[0].text(0.35, 0.05, '$\mathbf{M^T(R_e)}$: %4.2f'
                     % (MassRe['total']), transform=axes[0].transAxes,
                     fontproperties=text_font)
        axes[0].text(0.05, 0.25, '$\mathbf{M^*/L}$: %4.2f' % (self.ml),
                     transform=axes[0].transAxes, fontproperties=text_font)
        if MassRe['fdm'] is not None:
            axes[0].text(0.35, 0.25, '$\mathbf{f_{DM}(R_e)}$: %4.2f'
                         % (MassRe['fdm']), transform=axes[0].transAxes,
                         fontproperties=text_font)
        if MassRe['fdm'] is not None:
            axes[1].text(0.85, 0.05, 'Total', color='c',
                         transform=axes[1].transAxes, fontproperties=text_font)

            axes[1].text(0.85, 0.15, 'Dark', color='r',
                         transform=axes[1].transAxes, fontproperties=text_font)
            axes[1].text(0.85, 0.25, 'Stellar', color='y',
                         transform=axes[1].transAxes, fontproperties=text_font)
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
