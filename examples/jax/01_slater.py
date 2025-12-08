import pyscf.pbc.gto as gto
import pyscf.gto as molgto
import pyscf.pbc.dft as dft
import numpy
import pandas as pd


def make_cell_basis():
    """
    Here we make an uncontracted basis for the cell.
    Our JAX implementation requires this for efficient evaluation,
    and often for solids the uncontracted basis is much better because the
    others
    """
    cell = gto.Cell()
    cell.atom = '''C     0.      0.      0.    
                  C     0.8917  0.8917  0.8917
                  C     1.7834  1.7834  0.    
                  C     2.6751  2.6751  0.8917
                  C     1.7834  0.      1.7834
                  C     2.6751  0.8917  2.6751
                  C     0.      1.7834  1.7834
                  C     0.8917  2.6751  2.6751'''
    cell.basis = 'unc-ccecp-ccpvtz'
    cell.pseudo = 'ccecp' #These are high accuracy ECP's for QMC.
    cell.cart = True # also important for JAX efficiency.
    cell.a = numpy.eye(3)*3.5668
    cell.build()
    return cell



def run_dft(cell, chkfile):
    mf = dft.RKS(cell)
    mf.xc = 'lda,vwn'
    mf = mf.multigrid_numint()
    mf.chkfile = chkfile
    mf.kernel()
    return mf.e_tot


def generate_etb_set(cell, alpha0=0.2, l_polarization=2):
    """
    Generate an even tempered basis which is selected to best reproduce the
    basis in cell.

    You can use this as written for the first 3 rows and all sp elements.

    cell is a cell object with a given basis
    alpha0 is the longest range you would like to go. typically 0.2 or 0.1
    l_polarization is how many angular momentum functions you'd like to allow
    """
    #print(cell._basis)
    new_basis ={}
    for atomname, basis in cell._basis.items():
        maxl_contract = 1 + l_polarization
        # If you are doing a F-electron or 4d or 5d element then you
        # need to update this.
        if atomname in ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']:
            maxl_contract = 2+l_polarization

        # in this loop find the minimum and maximum exponents
        # for each angular momentum channel.
        maxexp = numpy.zeros(maxl_contract+1)
        minexp = numpy.ones(maxl_contract+1)*1000
        for element in basis:
            #print(element)
            l = element[0]
            exponents = numpy.max([e[0] for e in element[1:]])
            #print(exponents)
            if l <= maxl_contract:
                maxexp[l] = numpy.max([exponents, maxexp[l]])
                minexp[l] = numpy.min([exponents, minexp[l]])

        # Truncate the minimum exponent to alpha0
        minexp[minexp < alpha0] = alpha0
        # Now
        etbs = []
        for l, maxe in enumerate(maxexp):
            n = int(numpy.log2(maxe/minexp[l]))+2
            etbs.append((l, n, minexp[l], 2))
        new_basis[atomname] = molgto.etbs(etbs)
    newcell = cell.copy()
    newcell.basis = new_basis
    newcell.build()
    return newcell


if __name__ == "__main__":
    cell_orig = make_cell_basis()
    cell_etb = generate_etb_set(cell_orig, alpha0=0.2, l_polarization=2)
    cell_exp2 = cell_orig.copy()
    cell_exp2.exp_to_discard = 0.2
    cell_exp2.build()

    # Sometimes an even tempered basis is better than the uncontract and expand, and
    # sometimes not.
    print("etb basis", run_dft(cell_etb, 'etb0.2.hdf5'))
    print("uncontracted + exp_to_discard", run_dft(cell_exp2, 'exp_to_discard0.2.hdf5'))

    # You should probably also try decreasing alpha0, depending on the material.

