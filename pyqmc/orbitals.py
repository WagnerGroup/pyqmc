import numpy as np
import pyqmc.pbc as pbc

from pyqmc.supercell import get_supercell_kpts, get_supercell

def get_wrapphase_real(x):
    return (-1) ** np.round(x / np.pi)


def get_wrapphase_complex(x):
    return np.exp(1j * x)


def get_complex_phase(x):
    return x / np.abs(x)



class MoleculeOrbitalEvaluator:
    def __init__(self, mol, mo_coeff):
        self.iscomplex=False
        self.parameters={'mo_coeff':mo_coeff}
        self._mol = mol

    def aos(self, eval_str, configs, mask=None):
        """
        """
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        return self._mol.eval_gto(eval_str, mycoords)

    def mos(self, ao):
        return ao.dot(self.parameters['mo_coeff'])

    def pgradient(self, ao):
        return np.array([self.parameters['mo_coeff'].shape[1]]),ao
    
    
def get_k_indices(cell, mf, kpts, tol=1e-6):
    """Given a list of kpts, return inds such that mf.kpts[inds] is a list of kpts equivalent to the input list"""
    kdiffs = mf.kpts[np.newaxis] - kpts[:, np.newaxis]
    frac_kdiffs = np.dot(kdiffs, cell.lattice_vectors().T) / (2 * np.pi)
    kdiffs = np.mod(frac_kdiffs + 0.5, 1) - 0.5
    return np.nonzero(np.linalg.norm(kdiffs, axis=-1) < tol)[1]

class PBCOrbitalEvaluatorKpoints:
    """
    Evaluate orbitals from a 
    cell is expected to be one made with make_supercell().
    mo_coeff should be in [k][ao,mo] order
    kpts should be a list of the k-points corresponding to mo_coeff    
    """
    def __init__(self, cell, mo_coeff, kpts=None, S = None):
        self.iscomplex=True
        self._cell = cell.original_cell
        self.S = cell.S

        self._kpts = [0,0,0] if kpts is None else kpts 
        self.param_split = np.cumsum([m.shape[1] for m in mo_coeff])
        self.parameters={'mo_coeff': np.concatenate(mo_coeff, axis=1)}

    @classmethod
    def from_mean_field(self, cell, mf, twist=None, spin = None):
        """
        mf is expected to be a KUHF, KRHF, or equivalent DFT objects. 
        Selects occupied orbitals from a given twist and spin.
        If cell is a supercell, will automatically choose the folded k-points that correspond to that twist.
        """

        cell = cell if hasattr(cell, 'original_cell') else get_supercell(cell,np.asarray([[1,0,0],[0,1,0],[0,0,1]]))

        if twist is None:
            twist = np.zeros(3)
        else:
            twist = np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
        kinds = list(set(get_k_indices(cell, mf, get_supercell_kpts(cell) + twist)))
        if len(kinds) != cell.scale:
            raise ValueError("Did not find the right number of k-points for this supercell")
        kpts = mf.kpts[kinds]
        if spin is None:
            mo_coeff = [mf.mo_coeff[k] for k in kinds]
        else:
            mo_coeff = [mf.mo_coeff[spin][k][:,mf.mo_occ[spin][k]>0.5] for k in kinds]
        return PBCOrbitalEvaluatorKpoints(cell, mo_coeff, kpts)
        
    def aos(self,eval_str,configs, mask=None):
        """
        Returns an ndarray in order [k,coordinate, orbital] of the ao's
        """
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))

        #coordinate, dimension
        wrap = configs.wrap if mask is None else configs.wrap[mask]
        wrap = np.dot(wrap, self.S)
        wrap = wrap.reshape((-1,wrap.shape[-1]))
        kdotR = np.linalg.multi_dot(
            (self._kpts, self._cell.lattice_vectors().T, wrap.T)
        )
        # k, coordinate
        wrap_phase = get_wrapphase_complex(kdotR)

        # k,coordinate, orbital
        ao = np.asarray(self._cell.eval_gto("PBC"+eval_str, mycoords, kpts=self._kpts))
        print('ao shape',ao.shape)
        return np.einsum("ij,ijk->ijk",wrap_phase, ao)

        
    def mos(self, ao):
        """ao should be [k,coordinate,ao].
        Returns a concatenated list of all molecular orbitals in form [coordinate, mo]
        """
        # do some split
        p = np.split(self.parameters['mo_coeff'], self.param_split, axis=-1)
        return np.concatenate([ak.dot(mok) for ak,mok in zip(ao,p)], axis=-1)

    def pgradient(self,ao):
        """
        returns:
        N sets of atomic orbitals
        split: which molecular orbitals correspond to which set

        You can construct the determinant by doing, for example:
        split, aos = pgradient(self.aos)
        mos = np.split(range(nmo),split)
        for ao, mo in zip(aos,mos):
            for i in mo:
                pgrad[:,:,i] = self._testcol(i,spin,ao)
                
        """
        return self.param_split, ao
