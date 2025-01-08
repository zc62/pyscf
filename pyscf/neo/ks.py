#!/usr/bin/env python

'''
Non-relativistic Kohn-Sham for NEO-DFT
'''

import numpy
from pyscf import dft, lib
from pyscf.lib import logger
from pyscf.dft.numint import (BLKSIZE, NBINS, eval_ao, eval_rho, _scale_ao,
                              _dot_ao_ao, _dot_ao_ao_sparse)
from pyscf.neo import hf


def eval_epc(epc, rho_e, rho_n):
    '''Evaluate EPC energy and potentials for both components.

    Args:
        epc : str or dict
            EPC functional specification. Can be '17-1', '17-2', '18-1', '18-2'
            or a dict with 'epc_type' and parameters.
        rho_e : ndarray
            Electron density on grid points
        rho_n : ndarray
            Nuclear density on grid points

    Returns:
        exc : ndarray
            EPC energy density
        vxc_n : ndarray
            Nuclear potential
        vxc_e : ndarray
            Electronic potential
    '''
    params = {
        '17-1': (2.35, 2.4, 3.2),
        '17-2': (2.35, 2.4, 6.6),
        '18-1': (1.8, 0.1, 0.03),
        '18-2': (3.9, 0.5, 0.06)
    }
    # Parse EPC type and parameters
    if isinstance(epc, dict):
        epc_type = epc.get('epc_type', '17-2')
        if epc_type in ('17', '18'):
            a = epc['a']
            b = epc['b']
            c = epc['c']
        else:
            if epc_type not in params:
                raise ValueError(f'Unknown EPC type: {epc_type}')
            a, b, c = params[epc_type]
    else:
        epc_type = epc
        if epc_type not in params:
            raise ValueError(f'Unknown EPC type: {epc_type}')
        a, b, c = params[epc_type]

    # Evaluate energy and potentials based on functional type
    if epc_type.startswith('17'):
        # EPC17 form
        rho_prod = numpy.multiply(rho_e, rho_n)
        rho_sqrt = numpy.sqrt(rho_prod)
        denom = a - b * rho_sqrt + c * rho_prod
        denom2 = numpy.square(denom)

        # Energy density
        exc = -rho_e / denom

        # Nuclear potential
        numer_n = -a * rho_e + 0.5 * b * rho_e * rho_sqrt
        vxc_n = numer_n / denom2

        # Electronic potential
        numer_e = -a * rho_n + 0.5 * b * rho_n * rho_sqrt
        vxc_e = numer_e / denom2
    else:
        # EPC18 form
        rho_e_cbrt = numpy.cbrt(rho_e)
        rho_n_cbrt = numpy.cbrt(rho_n)
        beta = rho_e_cbrt + rho_n_cbrt
        beta2 = numpy.square(beta)
        beta3 = beta * beta2
        beta5 = beta2 * beta3
        beta6 = beta3 * beta3
        denom = a - b * beta3 + c * beta6
        denom2 = numpy.square(denom)

        # Energy density
        exc = -rho_e / denom

        # Nuclear potential
        numer_n = a * rho_e - b * rho_e_cbrt**4 * beta2 \
                + c * numpy.multiply(rho_e * beta5, rho_e_cbrt - rho_n_cbrt)
        vxc_n = -numer_n / denom2

        # Electronic potential
        numer_e = a * rho_n - b * rho_n_cbrt**4 * beta2 \
                + c * numpy.multiply(rho_n * beta5, rho_n_cbrt - rho_e_cbrt)
        vxc_e = -numer_e / denom2

    return exc, vxc_n, vxc_e

class InteractionCorrelation(hf.InteractionCoulomb):
    '''Inter-component Coulomb and correlation'''
    def __init__(self, *args, epc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epc = epc

    def _need_epc(self):
        if self.epc is None:
            return False
        if self.mf1_type == 'e':
            if self.mf2_type.startswith('n'):
                if self.mf2.mol.super_mol.atom_pure_symbol(self.mf2.mol.atom_index) == 'H':
                    if isinstance(self.epc, str) or \
                            self.mf2.mol.atom_index in self.epc['epc_nuc']:
                        return True
        if self.mf2_type == 'e':
            if self.mf1_type.startswith('n'):
                if self.mf1.mol.super_mol.atom_pure_symbol(self.mf1.mol.atom_index) == 'H':
                    if isinstance(self.epc, str) or \
                            self.mf1.mol.atom_index in self.epc['epc_nuc']:
                        return True
        return False

    def get_vint(self, dm, *args, **kwargs):
        vj = super().get_vint(dm, *args, **kwargs)
        # For nuclear initial guess, use Coulomb only
        if not (self.mf1_type in dm and self.mf2_type in dm and self._need_epc()):
            return vj

        if self.mf1_type == 'e':
            mf_e, dm_e = self.mf1, dm[self.mf1_type]
            mf_n, dm_n = self.mf2, dm[self.mf2_type]
            n_type = self.mf2_type
        else:
            mf_e, dm_e = self.mf2, dm[self.mf2_type]
            mf_n, dm_n = self.mf1, dm[self.mf1_type]
            n_type = self.mf1_type

        grids = mf_e.grids
        ni = mf_e._numint

        nao_e = mf_e.mol.nao
        nao_n = mf_n.mol.nao
        ao_loc_e = mf_e.mol.ao_loc_nr()
        ao_loc_n = mf_n.mol.ao_loc_nr()

        exc_sum = 0
        vxc_e = numpy.zeros((nao_e, nao_e))
        vxc_n = numpy.zeros((nao_n, nao_n))

        if dm_e.ndim > 2:
            dm_e = dm_e[0] + dm_e[1]

        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask_e = mf_e.mol.get_overlap_cond() < -numpy.log(ni.cutoff)

        non0tab_n = ni.make_mask(mf_n.mol, grids.coords, grids.weights.size)

        aow = None
        p1 = 0
        for ao_e, mask_e, weight, coords in ni.block_loop(mf_e.mol, grids, nao_e):
            p0, p1 = p1, p1 + weight.size
            mask_n = non0tab_n[p0//BLKSIZE:p1//BLKSIZE+1]

            # Skip if no nuclear basis functions in this block
            if numpy.all(mask_n == 0):
                continue

            rho_e = eval_rho(mf_e.mol, ao_e, dm_e, mask_e)
            rho_e[rho_e < 0] = 0  # Ensure non-negative density

            ao_n = eval_ao(mf_n.mol, coords, non0tab=mask_n)
            rho_n = eval_rho(mf_n.mol, ao_n, dm_n)
            rho_n[rho_n < 0] = 0  # Ensure non-negative density

            exc, vxc_n_grid, vxc_e_grid = eval_epc(self.epc, rho_e, rho_n)

            den = rho_n * weight
            exc_sum += numpy.dot(den, exc)

            # x0.5 for vmat + vmat.T
            aow = _scale_ao(ao_n, 0.5 * weight * vxc_n_grid, out=aow)
            vxc_n += _dot_ao_ao(mf_n.mol, ao_n, aow, mask_n,
                                (0, mf_n.mol.nbas), ao_loc_n)
            _dot_ao_ao_sparse(ao_e, ao_e, 0.5 * weight * vxc_e_grid,
                              nbins, mask_e, pair_mask_e, ao_loc_e, 1, vxc_e)

        vxc_n = vxc_n + vxc_n.conj().T
        vxc_e = vxc_e + vxc_e.conj().T

        vxc = {}
        vxc['e'] = lib.tag_array(vj['e'] + vxc_e, exc=exc_sum, vj=vj['e'])
        vxc[n_type] = lib.tag_array(vj[n_type] + vxc_n, exc=exc_sum, vj=vj[n_type])
        return vxc

class KS(hf.HF):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.KS(mol, xc='b3lyp5', epc='17-2')
    >>> mf.max_cycle = 100
    >>> mf.scf()
    -100.38833734158459
    '''

    def __init__(self, mol, xc, *args, epc=None, **kwargs):
        # NOTE: To prevent user error, require xc to be explicitly provided
        super().__init__(mol, *args, **kwargs)
        self.xc_e = xc # Electron xc functional
        self.epc = epc # Electron-proton correlation

        for t, comp in self.mol.components.items():
            if not t.startswith('n'):
                if self.unrestricted:
                    mf = dft.UKS(comp, xc=self.xc_e)
                else:
                    mf = dft.RKS(comp, xc=self.xc_e)
                if self.df_ee:
                    mf = mf.density_fit(auxbasis=self.auxbasis_e, only_dfj=self.only_dfj_e)
                charge = 1.
                if t.startswith('p'):
                    charge = -1.
                self.components[t] = hf.general_scf(mf, charge=charge)
        self.interactions = hf.generate_interactions(self.components, InteractionCorrelation,
                                                     self.max_memory, epc=self.epc)

    def energy_elec(self, dm=None, h1e=None, vhf=None, vint=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        if vint is None: vint = self.get_vint(self.mol, dm)
        self.scf_summary['e1'] = 0
        self.scf_summary['coul'] = 0
        self.scf_summary['exc'] = 0
        e_elec = 0
        e2 = 0
        for t, comp in self.components.items():
            logger.debug(self, f'Component: {t}')
            # Assign epc correlation energy to electrons
            if hasattr(vhf[t], 'exc') and hasattr(vint[t], 'exc'):
                vhf[t].exc += vint[t].exc
            if hasattr(vint[t], 'vj'):
                vj = vint[t].vj
            else:
                vj = vint[t]
            # vj acts as if a spin-insensitive one-body Hamiltonian
            # .5 to remove double-counting
            e_elec_t, e2_t = comp.energy_elec(dm[t], h1e[t] + vj * .5, vhf[t])
            e_elec += e_elec_t
            e2 += e2_t
            self.scf_summary['e1'] += comp.scf_summary['e1']
            # Nucleus is RHF and its scf_summary does not have coul or exc
            if hasattr(vhf[t], 'exc'):
                self.scf_summary['coul'] += comp.scf_summary['coul']
                self.scf_summary['exc'] += comp.scf_summary['exc']
        return e_elec, e2

    def get_vint(self, mol=None, dm=None):
        '''Inter-type Coulomb and possible epc'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vint = {}
        for t in self.components.keys():
            vint[t] = 0
        for t_pair, interaction in self.interactions.items():
            v = interaction.get_vint(dm)
            for t in t_pair:
                # Take care of tag_array, accumulate exc and vj
                v_has_tag = hasattr(v[t], 'exc')
                vint_has_tag = hasattr(vint[t], 'exc')
                if v_has_tag:
                    if vint_has_tag:
                        exc = vint[t].exc + v[t].exc
                        vj = vint[t].vj + v[t].vj
                    else:
                        exc = v[t].exc
                        vj = v[t].vj
                    vint[t] = lib.tag_array(vint[t] + v[t], exc=exc, vj=vj)
                else:
                    if vint_has_tag:
                        vint[t] = lib.tag_array(vint[t] + v[t], exc=vint[t].exc, vj=vint[t].vj + v[t])
                    else:
                        vint[t] += v[t]
        return vint

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        old_keys = sorted(self.components.keys())
        super().reset(mol=mol)
        if old_keys != sorted(self.components.keys()):
            # quantum nuc is different, need to rebuild
            for t, comp in self.mol.components.items():
                if not t.startswith('n'):
                    if self.unrestricted:
                        mf = dft.UKS(comp, xc=self.xc_e)
                    else:
                        mf = dft.RKS(comp, xc=self.xc_e)
                    if self.df_ee:
                        mf = mf.density_fit(auxbasis=self.auxbasis_e, only_dfj=self.only_dfj_e)
                    charge = 1.
                    if t.startswith('p'):
                        charge = -1.
                    self.components[t] = general_scf(mf, charge=charge)
            self.interactions = generate_interactions(self.components, InteractionCorrelation,
                                                      self.max_memory, epc=self.epc)
        return self
