import numpy as np
from pyscf import lib, gto, scf

class rnExp():
    def __init__(self,n,e,c):
        self.n=np.asarray(n)
        self.e=np.asarray(e)
        self.c=np.asarray(c)

    def __call__(self,r):
        return np.sum(r[:,np.newaxis]**self.n * self.c*np.exp(-self.e*r[:,np.newaxis]**2),axis=1)

def generate_ecp_functors(coeffs):
    d={}
    for c in coeffs:
        el=c[0]
        rn=[]
        exponent=[]
        coefficient=[]
        for n,expand in enumerate(c[1]):
           #print("r",n-2,"coeff",expand)
           for line in expand:
               rn.append(n)
               exponent.append(line[0])
               coefficient.append(line[1])
        d[el]=rnExp(rn,exponent,coefficient)
    return d


#########################################################################

def P_l(x,l): # legendre polynomial, x=r_ea(i)
    # returns a nconf x naip array for a given l
    if l == 0:  return np.ones(x.shape)
    elif l == 1: return x
    elif l == 2: return 0.5*(3*x*x-np.ones(x.shape))
    elif l == 3: return 0.5*(5*x*x*x-3*x)
    elif l == 4: return 0.125*(35*x*x*x*x- 30*x*x+3*np.ones(x.shape))
    else: return np.zeros(x.shape)

def get_r_ea(mol,configs,e,at):
    # returns a nconf x 3 array, distance between electron e and atom at
    epos = configs[:,e,:]
    nconf = configs.shape[0]
    apos = np.outer(np.ones(nconf),np.array(mol._atom[at][1])) # nconf x 3 array, position of atom at
    return epos-apos

def get_r_ea_i(mol,configs_rot,e,at):
    # configs_rot is the rotated electron positions: nconf x naip x nelec x 3
    nconf,naip = configs_rot.shape[0:2]
    apos = np.zeros([nconf,naip,3]) # position of the atom, broadcasted into nconf x naip x 3
    for aip in range(naip):
        apos[:,aip,:] = np.outer(np.ones(nconf),np.array(mol._atom[at][1]))
    epos_rot = configs_rot[:,:,e,:]
    return epos_rot-apos


def get_v_l(mol,configs,e,at):
    # returns a nconf x nl array
    nconf = configs.shape[0]
    at_name = mol._atom[at][0]
    r_ea = np.linalg.norm(get_r_ea(mol,configs,e,at),axis = 1)  # returns a nconf array of the electron-atom dist 
    vl = generate_ecp_functors(mol._ecp[at_name][1])
    Lmax = len(vl)
    v_l = np.zeros([nconf,Lmax])
    for l in vl.keys(): # -1,0,1,...
        v_l[:,l] = vl[l](r_ea)
    return vl.keys(),v_l   # nconf x nl


def get_wf_ratio(wf,configs_rot,e):
    # returns the Psi(r_e(i))/Psi(r_e) value, which is a nconf x naip array
    nconf,naip = configs_rot.shape[0:2]
    wf_ratio = np.zeros([nconf,naip])
    for aip in range(naip):
        epos = configs_rot[:,aip,e,:].reshape([nconf,3])
        wf_ratio[:,aip] = wf.testvalue(e,epos)
    return wf_ratio

def get_P_l(mol,configs,weights,configs_rot,l_list,e,at):
    at_name = mol._atom[at][0]
    nconf,naip = configs_rot.shape[0:2]
    
    P_l_val = np.zeros([nconf,naip,len(l_list)])
    r_ea = get_r_ea(mol,configs,e,at)  # nconf x 3
    r_ea_i = get_r_ea_i(mol,configs_rot,e,at) # nconf x naip x 3
    rdotR = np.zeros(r_ea_i.shape[0:2])  # nconf x naip

    # get the cosine values
    for aip in range(naip):  
        rdotR[:,aip] = r_ea[:,0]*r_ea_i[:,aip,0] + r_ea[:,1]*r_ea_i[:,aip,1] + r_ea[:,2]*r_ea_i[:,aip,2]
        rdotR[:,aip] /= np.linalg.norm(r_ea,axis=1)*np.linalg.norm(r_ea_i[:,aip,:],axis=1)
    #print('cosine values',rdotR)

    # already included the factor (2l+1), and the quadrature weights here
    for l in l_list:
        P_l_val[:,:,l] = (2*l+1)*P_l(rdotR,l)*np.outer(np.ones(nconf),weights) 
    return P_l_val  # nconf x naip x nl

#########################################################################

def ecp_ea(mol,configs,wf,e,at):
    weights,configs_rot = get_rot(mol,configs,e,at,naip=6)
    l_list,v_l = get_v_l(mol,configs,e,at)
    P_l = get_P_l(mol,configs,weights,configs_rot,l_list,e,at)
    ratio = get_wf_ratio(wf,configs_rot,e)
    ecp_val = np.einsum("ij,ik,ijk->i", ratio, v_l, P_l)
    # compute the local part
    local_l = -1
    ecp_val += v_l[:,local_l]
    return ecp_val   

def ecp(mol,configs,wf):
    if mol._ecp != {}:
        nconf,nelec = configs.shape[0:2]
        ecp_tot = np.zeros(nconf)
    for e in range(nelec):
        for at in range(np.shape(mol._atom)[0]):
            ecp_tot += ecp_ea(mol,configs,wf,e,at)
    return ecp_tot

#################### Quadrature Rules ############################
def get_angles(t,p,naip=6):  # currently only naip = 6 is supported
    d1 = {}
    d1[0] = t
    d1[1] = np.pi*np.ones(t.shape)-t
    d1[2] = 0.5*np.pi*np.ones(t.shape)
    d1[3] = 0.5*np.pi*np.ones(t.shape)
    d1[4] = 0.5*np.pi*np.ones(t.shape)+t
    d1[5] = 0.5*np.pi*np.ones(t.shape)-t
    d2 = {}
    d2[0] = p
    d2[1] = np.pi*np.ones(t.shape)+p
    d2[2] = -0.5*np.pi*np.ones(t.shape)+p
    d2[3] = 0.5*np.pi*np.ones(t.shape)+p
    d2[4] = p
    d2[5] = np.pi*np.ones(t.shape)+p
    return d1,d2

def get_rot(mol,configs,e,at,naip=6):
    nconf,nelec = configs.shape[0:2]
    apos = np.outer(np.ones(nconf),np.array(mol._atom[at][1]))

    r_ea_vec = get_r_ea(mol,configs,e,at)
    r_ea = np.linalg.norm(r_ea_vec,axis = 1)

    theta = np.arccos(r_ea_vec[:,2]/r_ea)
    phi = np.arctan(r_ea_vec[:,1]/r_ea_vec[:,0])

    t,p = get_angles(theta,phi,naip)

    configs_rot = np.zeros([nconf,naip,nelec,3])
    for aip in range(naip):
        configs_rot[:,aip,:,:] = configs # init
        configs_rot[:,aip,e,0] = -apos[:,0]+r_ea*np.sin(t[aip])*np.cos(p[aip])
        configs_rot[:,aip,e,1] = -apos[:,1]+r_ea*np.sin(t[aip])*np.sin(p[aip])
        configs_rot[:,aip,e,2] = -apos[:,2]+r_ea*np.cos(t[aip])
    weights = 1./naip*np.ones(naip)
    return weights,configs_rot

from slateruhf import PySCFSlaterUHF
def test():
    mol = gto.M(atom='C 0. 0. 0.',ecp='bfd',basis = 'bfd_vtz')
    mf = scf.UHF(mol).run()

    nconf=10
    nelec=np.sum(mol.nelec)

    slater=PySCFSlaterUHF(nconf,mol,mf)

    configs=np.random.randn(nconf,nelec,3)

    import testwf
    print("testing internals:", testwf.test_updateinternals(slater,configs))

    ecp_val = ecp(mol,configs,slater)
    print("ecp nonlocal values:", ecp_val)

if __name__=="__main__":
    test()

