import numpy as np
""" 
Collection of 3d functions
"""

class GaussianFunction:
    def __init__(self,exponent):
        self.parameters={}
        self.parameters['exponent']=exponent

    def value(self,x): 
        """Return the value of the function. 
        x should be a (nconfig,3) vector """
        r2=np.sum(x**2,axis=1)
        return np.exp(-self.parameters['exponent']*r2)
        
    def gradient(self,x):
        """ return gradient of the function """
        v=self.value(x)
        return -2*self.parameters['exponent']*x*v[:,np.newaxis]

    def laplacian(self,x):
        """ laplacian """
        v=self.value(x)
        alpha=self.parameters['exponent']
        return (4*alpha*alpha*x*x-2*alpha)*v[:,np.newaxis]


    def pgradient(self,x):
        """ parameter gradient """
        
class PadeFunction:
    """
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    alpha_k = alpha/2^k, k starting at 0
    """
    def __init__(self, alphak):
        self.parameters={}
        self.parameters['alphak'] = alphak

    def value(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3 (number of inner dimensions doesn't matter)
        Return:
          func: same dimensions as rvec, but the last one removed 
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = self.parameters['alphak']* r
        return (a/(1+a))**2

    def gradient(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3, displacement between particles
            For example, nconf x n_elec_pairs x 3, where n_elec_pairs could be all pairs of electrons or just the pairs that include electron e for the purpose of updating one electron.
            Or it could be nconf x nelec x natom x 3 for electron-ion displacements
        Return:
          grad: same dimensions as rvec
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        a = self.parameters['alphak']* r
        grad = 2* self.parameters['alphak']**2/(1+a)**3 *rvec
        return grad
        
    def laplacian(self, rvec):
        """
        Parameters:
          rvec: nconf x ... x 3
        Return:
          lap: same dimensions as rvec, d2/dx2, d2/dy2, d2/dz2 separately
        """
        r = np.linalg.norm(rvec, axis=-1, keepdims=True)
        a = self.parameters['alphak']* r
        #lap = 6*self.parameters['alphak']**2 * (1+a)**(-4) #scalar formula
        lap = 2*self.parameters['alphak']**2 * (1+a)**(-3) \
              *(1 - 3*a/(1+a)*(rvec/r)**2)
        return lap

    def pgradient(self, rvec):
        """ Return gradient of value with respect to parameter alphak
        Parameters:
          rvec: nconf x ... x 3
        Return:
          akderiv: same dimensions as rvec, but the last one removed 
        
        """
        r = np.linalg.norm(rvec, axis=-1)
        a = self.parameters['alphak']* r
        akderiv = 2*a/(1+a)**3 * r
        return akderiv

def test(): 
    import testwf 
    pade = PadeFunction(0.2)
    gauss = GaussianFunction(0.4)
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        print('Gaussian: delta', delta, "Testing gradient", testwf.test_func3d_gradient(gauss,delta=delta))
        print('Gaussian: delta', delta, "Testing laplacian", testwf.test_func3d_laplacian(gauss,delta=delta))
    for delta in [1e-3,1e-4,1e-5,1e-6,1e-7]:
        print('Pade: delta', delta, "Testing gradient", testwf.test_func3d_gradient(pade,delta=delta))
        print('Pade: delta', delta, "Testing laplacian", testwf.test_func3d_laplacian(pade,delta=delta))
    

def test_pade():
    pade = PadeFunction(0.2)
    parms = np.random.random(4)*2-1
    epos = np.random.random((1, 8, 3))
    atoms = np.array([[0,0,0],[0,0,1.5],[1,1,0]]) 
    rvec = epos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
    print('rvec', rvec.shape)
    val = pade.value(rvec)
    grad = pade.gradient(rvec)
    lap = pade.laplacian(rvec)
    delta=1e-5
    e=2
    testlap = 0
    for i in range(3):
      pos = epos.copy()
      pos[:,e,i] += delta
      plusvec = pos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
      plusval = pade.value(plusvec)
      pos[:,e,i] -= 2*delta
      minuvec = pos[:,:,np.newaxis,:] - atoms[np.newaxis, np.newaxis,:,:]
      minuval = pade.value(minuvec)
      deriv = (plusval - minuval)/(2*delta)
      testlap += (plusval + minuval - 2*val)/delta**2
      print(i, np.linalg.norm(grad[:,e,:,i]-deriv[:,e,:]))
      print(i, grad[:,e,:,i])
      print(i, deriv[:,e,:])
    print('lap', np.linalg.norm(lap[:,e,:] - testlap[:,e,:]))

if __name__=="__main__":
    test()
