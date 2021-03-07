import numpy as np
import pyqmc.pbc as pbc
from pyqmc.supercell import get_supercell_kpts, get_supercell
import pyqmc.determinant_tools

def get_wrapphase_real(x):
    return (-1) ** np.round(x / np.pi)

def get_wrapphase_complex(x):
    return np.exp(1j * x)

def get_complex_phase(x):
    return x / np.abs(x)

def choose_evaluator_from_pyscf(mol, mf, mc=None, twist=None):
    """
    Returns:
    a molecular orbital evaluator

    """

    if hasattr(mol, "a"): 
        if mc is not None:
            raise NotImplementedError("Do not support multiple determinants for k-points orbital evaluator")
        return PBCOrbitalEvaluatorKpoints.from_mean_field(mol, mf, twist)
    if mc is None:
        return MoleculeOrbitalEvaluator.from_pyscf(mol, mf)
    return MoleculeOrbitalEvaluator.from_pyscf(mol, mf, mc)



"""
The evaluators have the concept of a 'set' of atomic orbitals, that may apply to 
different sets of molecular orbitals

For example, for the PBC evaluator, each k-point is a set, since each molecular 
orbital is only a sum over the k-point of its type.

In the future, this could apply to orbitals of a given point group symmetry, for example.

"""

class MoleculeOrbitalEvaluator:
    def __init__(self, mol, mo_coeff):
        self.iscomplex=False
        self.parameters={'mo_coeff_alpha':mo_coeff[0],
                        'mo_coeff_beta':mo_coeff[1]}
        self.parm_names=['_alpha','_beta']

        self._mol = mol

    @classmethod
    def from_pyscf(self, mol, mf, mc=None, tol=-1):
        """
        mol: A Mole object
        mf: An object with mo_coeff and mo_occ. 
        mc: (optional) a CI object from pyscf

        """
        obj = mc if hasattr(mc, 'mo_coeff') else mf
        if mc is not None:
            detcoeff, occup, det_map = pyqmc.determinant_tools.interpret_ci(mc, tol)
        else: 
            detcoeff = np.array([1.0])
            det_map = np.array([[0],[0]])
            #occup
            if len(mf.mo_occ.shape) == 2:
                occup = [[list(np.argwhere(mf.mo_occ[spin] > 0.5)[:,0])] for spin in [0,1]]
            else:
                occup = [[list(np.argwhere(mf.mo_occ > 1.5-spin)[:,0])] for spin in [0,1]]

        max_orb = [np.max(occup[s])+1 for s in [0,1]]

        if len(mf.mo_occ.shape) == 2:
            mo_coeff = [obj.mo_coeff[spin][:,0:max_orb[spin]] for spin in [0,1]]
        else:
            mo_coeff = [obj.mo_coeff[:,0:max_orb[spin]] for spin in [0,1]]

        return detcoeff, occup, det_map, MoleculeOrbitalEvaluator(mol, mo_coeff)


    def aos(self, eval_str, configs, mask=None):
        """
        
        """
        mycoords = configs.configs if mask is None else configs.configs[mask]
        mycoords = mycoords.reshape((-1, mycoords.shape[-1]))
        return np.asarray([self._mol.eval_gto(eval_str, mycoords)])

    def mos(self, ao, spin):
        return ao[0].dot(self.parameters[f'mo_coeff{self.parm_names[spin]}'])

    def pgradient(self, ao, spin):
        return np.array([self.parameters[f'mo_coeff{self.parm_names[spin]}'].shape[1]]),ao
    

    
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
    mo_coeff should be in [spin][k][ao,mo] order
    kpts should be a list of the k-points corresponding to mo_coeff  

    """
    def __init__(self, cell, mo_coeff, kpts=None, S = None):
        self.iscomplex=True
        self._cell = cell.original_cell
        self.S = cell.S

        self._kpts = [0,0,0] if kpts is None else kpts 
        self.param_split = [np.cumsum([m.shape[1] for m in mo_coeff[spin]]) for spin in [0,1]]
        self.parm_names=['_alpha','_beta']
        self.parameters={'mo_coeff_alpha': np.concatenate(mo_coeff[0], axis=1),
                         'mo_coeff_beta': np.concatenate(mo_coeff[1], axis=1)}

    @classmethod
    def from_mean_field(self, cell, mf, twist=None):
        """
        mf is expected to be a KUHF, KRHF, or equivalent DFT objects. 
        Selects occupied orbitals from a given twist 
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

        detcoeff = np.array([1.0])
        det_map = np.array([[0],[0]])

        if len(mf.mo_coeff[0][0].shape) == 2:
            occup_k = [[[list(np.argwhere(mf.mo_occ[spin][k] > 0.5)[:,0])] for k in kinds ] for spin in [0,1]]
        elif len(mf.mo_coeff[0][0].shape) == 1:
            occup_k = [[[list(np.argwhere(mf.mo_occp[k] > 1.5-spin)[:,0])] for k in kinds ] for spin in [0,1]]

        occup = [[],[]]
        for spin in [0,1]:
            for occ_k in occup_k[spin]:
                occup[spin] += occ_k

        kpts = mf.kpts[kinds]
        if len(mf.mo_coeff[0][0].shape) == 2:
            mo_coeff = [[mf.mo_coeff[spin][k][:,mf.mo_occ[spin][k]>0.5] for k in kinds] for spin in [0,1]]
        elif len(mf.mo_coeff[0][0].shape) == 1:
            mo_coeff = [[mf.mo_coeff[k] for k in kinds] for spin in [0,1]]
        else:
            raise ValueError("Did not expect an scf object of type", type(mf))

        return detcoeff, occup, det_map, PBCOrbitalEvaluatorKpoints(cell, mo_coeff, kpts)
        
    def aos(self,eval_str,configs, mask=None):
        """
        Returns an ndarray in order [k,coordinate, orbital] of the ao's if value is requested

        if a derivative is requested, will instead return [k,d,coordinate,orbital]
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
        #print('ao shape',ao.shape)
        return np.einsum("...,...k->...k",wrap_phase, ao)

        
    def mos(self, ao, spin):
        """ao should be [k,coordinate,ao].
        Returns a concatenated list of all molecular orbitals in form [coordinate, mo]

        In the derivative case, returns [d,coordinate, mo]
        """
        # do some split
        p = np.split(self.parameters[f'mo_coeff{self.parm_names[spin]}'], self.param_split[spin], axis=-1)
        return np.concatenate([ak.dot(mok) for ak,mok in zip(ao,p)], axis=-1)

    def pgradient(self,ao, spin):
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
        return self.param_split[spin], ao
