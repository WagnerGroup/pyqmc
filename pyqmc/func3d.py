import numpy as np

""" 
Collection of 3d function objects. Each has a dictionary parameters, which corresponds
to any variational parameters the funtion has.

They should implement the following functions, all of which take input value x.
x should be of dimension (nconf,...,3).

value(x):
    returns f(x)
gradient(x)
    returns grad f(x) (nconf,...,3)
laplacian(x)
    returns diagonals of Hessian (nconf,...,3)
pgradient(x)
    returns dp f(x) as a dictionary corresponding to the keys of self.parameters
"""


class GaussianFunction:
    r"""A representation of a Gaussian: 
    :math: `\exp(-\alpha r^2)`
    where :math: `\alpha` can be accessed through parameters['exponent']

    """

    def __init__(self, exponent):
        self.parameters = {}
        self.parameters["exponent"] = exponent

    def value(self, x, r):
        """Returns function exp(-exponent*r^2).
        Parameters:
          x: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          func: (nconfig,...) vector
        """
        return np.exp(-self.parameters["exponent"] * r * r)

    def gradient(self, x, r):
        """Returns gradient of function.
        Parameters:
          x: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          grad: (nconfig,...,3) vector
        """
        v = self.value(x, r)
        return -2 * self.parameters["exponent"] * x * v[..., np.newaxis]

    def laplacian(self, x, r):
        """Returns laplacian of function.
        Parameters:
          x: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          lap: (nconfig,...,3) vector (components of laplacian d^2/dx_i^2 separately)
        """
        v = self.value(x, r)
        alpha = self.parameters["exponent"]
        return (4 * alpha * alpha * x * x - 2 * alpha) * v[..., np.newaxis]

    def gradient_laplacian(self, x, r):
        """Returns gradient and laplacian of function.
        Parameters:
          x: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          grad, lap: (nconfig,...,3) vectors (components of laplacian d^2/dx_i^2 separately)
        """
        v = self.value(x, r)[..., np.newaxis]
        alpha = self.parameters["exponent"]
        grad = -2 * alpha * x * v
        lap = (4 * alpha * alpha * x * x - 2 * alpha) * v
        return grad, lap

    def pgradient(self, x, r):
        """Returns parameters gradient.
        Parameters:
          x: (nconfig,...,3) vector
        Returns:
          pgrad: dictionary {'exponent':d/dexponent}
        """
        r2 = r * r
        return {"exponent": -r2 * np.exp(-self.parameters["exponent"] * r2)}


class PadeFunction:
    """
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    alpha_k = alpha/2^k, k starting at 0
    """

    def __init__(self, alphak):
        self.parameters = {}
        self.parameters["alphak"] = alphak

    def value(self, rvec, r):
        """
        Parameters:
          rvec: nconf x ... x 3 (number of inner dimensions doesn't matter)
        Return:
          func: same dimensions as rvec, but the last one removed 
        """
        a = self.parameters["alphak"] * r
        return (a / (1 + a)) ** 2

    def gradient(self, rvec, r):
        """
        Parameters:
          rvec: nconf x ... x 3, displacement between particles
            For example, nconf x n_elec_pairs x 3, where n_elec_pairs could be all pairs of electrons or just the pairs that include electron e for the purpose of updating one electron.
            Or it could be nconf x nelec x natom x 3 for electron-ion displacements
        Return:
          grad: same dimensions as rvec
        """
        a = self.parameters["alphak"] * r[..., np.newaxis]
        grad = 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3 * rvec
        return grad

    def laplacian(self, rvec, r):
        """
        Parameters:
          rvec: nconf x ... x 3
        Return:
          lap: same dimensions as rvec, d2/dx2, d2/dy2, d2/dz2 separately
        """
        a = self.parameters["alphak"] * r[..., np.newaxis]
        # lap = 6*self.parameters['alphak']**2 * (1+a)**(-4) #scalar formula
        lap = (
            2
            * self.parameters["alphak"] ** 2
            * (1 + a) ** (-3)
            * (1 - 3 * a / (1 + a) * (rvec / r[..., np.newaxis]) ** 2)
        )
        return lap

    def gradient_laplacian(self, rvec, r):
        """Returns gradient and laplacian of function.
        Parameters:
          rvec: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          grad, lap: (nconfig,...,3) vectors (components of laplacian d^2/dx_i^2 separately)
        """
        a = self.parameters["alphak"] * r[..., np.newaxis]
        temp = 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3
        grad = temp * rvec
        lap = temp * (1 - 3 * a / (1 + a) * (rvec / r[..., np.newaxis]) ** 2)
        return grad, lap

    def pgradient(self, rvec, r):
        """ Return gradient of value with respect to parameter alphak
        Parameters:
          rvec: nconf x ... x 3
        Return:
          pgrad: dictionary {'alphak':d/dalphak} with akderiv dimensions (config,)
        """
        a = self.parameters["alphak"] * r
        akderiv = 2 * a / (1 + a) ** 3 * r
        return {"alphak": akderiv}


class PolyPadeFunction:
    """
    :math:`b(r) = \frac{1-p(z)}{1+\beta p(z)}`
    :math:`z = r/r_{\rm cut}`
    where 
    :math:`p(z) = 6z^2 - 8z^3 + 3z^4`
    This function is positive at small r, decreasing to zero at r=rcut, being cutoff to 
    zero for r>rcut.
    """

    def __init__(self, beta, rcut):
        self.parameters = {}
        self.parameters["beta"] = beta
        self.parameters["rcut"] = rcut

    def value(self, rvec, r):
        """Returns 
        Parameters:
          rvec: (nconf,...,3) 
          r: (nconf,...) 
              magnitude of rvec
        Returns:
          func: (1-p(r/rcut))/(1+beta*p(r/rcut))
        """
        z = r / self.parameters["rcut"]
        p = z * z * (6 - 8 * z + 3 * z * z)
        func = (1 - p) / (1 + self.parameters["beta"] * p)
        func[z > 1] = 0.0
        return func

    def gradient(self, rvec, r):
        """
        Parameters:
          rvec: (nconf,...,3) 
        Returns:
          grad: (nconf,...,3)
        """
        grad = np.zeros(rvec.shape)
        mask = r > self.parameters["rcut"]
        r = r[..., np.newaxis]
        z = r / self.parameters["rcut"]
        p = z * z * (6 - 8 * z + 3 * z * z)
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dbdp = -(1 + self.parameters["beta"]) / (1 + self.parameters["beta"] * p) ** 2
        dzdx = rvec / (r * self.parameters["rcut"])
        grad = dbdp * dpdz * dzdx
        grad[mask] = 0
        return grad

    def laplacian(self, rvec, r):
        """
        Parameters:
          rvec: (nconf,...,3) 
        Returns:
          lapl: (nconf,...,3) 
              returns components of laplacian d^2/dx_i^2 separately
        """
        lapl = np.zeros(rvec.shape)
        mask = r > self.parameters["rcut"]
        r = r[..., np.newaxis]
        rvec = rvec
        z = r / self.parameters["rcut"]
        beta = self.parameters["beta"]

        p = z * z * (6 - 8 * z + 3 * z * z)
        dbdp = -(1 + beta) / (1 + beta * p) ** 2
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dzdx = rvec / (r * self.parameters["rcut"])
        d2pdz2_over_dpdz = (3 * z - 1) / (z * (z - 1))
        d2bdp2_over_dbdp = -2 * beta / (1 + beta * p)
        d2zdx2 = (1 - (rvec / r) ** 2) / (r * self.parameters["rcut"])
        grad = dbdp * dpdz * dzdx
        lapl = grad * (d2bdp2_over_dbdp * dpdz * dzdx + d2pdz2_over_dpdz * dzdx)
        lapl += dbdp * dpdz * d2zdx2
        lapl[mask] = 0
        return lapl

    def gradient_laplacian(self, rvec, r):
        """Returns gradient and laplacian of function.
        Parameters:
          rvec: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          grad, lap: (nconfig,...,3) vectors (components of laplacian d^2/dx_i^2 separately)
        """
        mask = r > self.parameters["rcut"]
        r = r[..., np.newaxis]
        rvec = rvec
        z = r / self.parameters["rcut"]
        beta = self.parameters["beta"]

        p = z * z * (6 - 8 * z + 3 * z * z)
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dbdp = -(1 + beta) / (1 + beta * p) ** 2
        dzdx = rvec / (r * self.parameters["rcut"])
        grad = dbdp * dpdz * dzdx
        d2pdz2_over_dpdz = (3 * z - 1) / (z * (z - 1))
        d2bdp2_over_dbdp = -2 * beta / (1 + beta * p)
        d2zdx2 = (1 - (rvec / r) ** 2) / (r * self.parameters["rcut"])
        lap = grad * (d2bdp2_over_dbdp * dpdz * dzdx + d2pdz2_over_dpdz * dzdx)
        lap += dbdp * dpdz * d2zdx2
        grad[mask] = 0
        lap[mask] = 0
        return grad, lap

    def pgradient(self, rvec, r):
        """ Returns gradient of self.value with respect to all parameters
        Parameters:
          rvec: (nconf,...,3) 
          rvec: (nconf,...) 
        Returns:
          paramderivs: dictionary {'rcut':d/drcut,'beta':d/dbeta}
        """
        pderiv = {"rcut": np.zeros(r.shape), "beta": np.zeros(r.shape)}
        mask = r <= self.parameters["rcut"]
        r = r[mask]
        z = r / self.parameters["rcut"]
        beta = self.parameters["beta"]

        p = z * z * (6 - 8 * z + 3 * z * z)
        dbdp = -(1 + beta) / (1 + beta * p) ** 2
        dpdz = 12 * z * (z * z - 2 * z + 1)
        pderiv["rcut"][mask] = dbdp * dpdz * (-z / self.parameters["rcut"])
        pderiv["beta"][mask] = -p * (1 - p) / (1 + beta * p) ** 2
        return pderiv


class CutoffCuspFunction:
    r"""
    :math:`b(r) = -\frac{p(r/r_{cut})}{1+\gamma*p(r/r_{cut})} + \frac{1}{3+\gamma}` 
    where 
    :math:`p(y) = y - y^2 + y^3/3`
    This function is positive at small r, decreasing to zero at r=rcut, being cutoff to 
    zero for r>rcut.
    """

    def __init__(self, gamma, rcut):
        self.parameters = {}
        self.parameters["gamma"] = gamma
        self.parameters["rcut"] = rcut

    def value(self, rvec, r):
        """Returns 
        Parameters:
          rvec: (nconf,...,3) vector
        Returns:
          func: p(r/rcut)/(1+gamma*p(r/rcut))
        """
        y = r / self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        p = y - y * y + y * y * y / 3
        func = -p / (1 + gamma * p) + 1 / (3 + gamma)
        func[y > 1] = 0.0
        return func * self.parameters["rcut"]

    def gradient(self, rvec, r):
        """
        Parameters:
          rvec: (nconf,...,3) vector
        Returns:
          grad: has same dimensions as rvec 
        """
        grad = np.zeros(rvec.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r > rcut
        r = r[..., np.newaxis]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = a / (1 + gamma * b) ** 2 / (rcut * r)

        grad = -rvec * c * rcut
        grad[mask] = 0
        return grad

    def laplacian(self, rvec, r):
        """
        Parameters:
          rvec: (nconf,...,3) vector
        Returns:
          lapl: has same dimensions as rvec, because returns components of laplacian d^2/dx_i^2 separately
        """
        lap = np.zeros(rvec.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r > rcut
        r = r[..., np.newaxis]
        rvec = rvec
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        temp = 2 * (y - 1) / (rcut * r)
        temp -= a / r ** 2
        temp -= 2 * a * a * c * gamma * (1 + gamma * b)
        lap = -rcut * c * (a + rvec ** 2 * temp)
        lap[mask] = 0
        return lap

    def gradient_laplacian(self, rvec, r):
        """Returns gradient and laplacian of function.
        Parameters:
          rvec: (nconfig,...,3) vector
          r: (nconfig,...) vector
        Returns:
          grad, lap: (nconfig,...,3) vectors (components of laplacian d^2/dx_i^2 separately)
        """
        grad = np.zeros(rvec.shape)
        lap = np.zeros(rvec.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r > rcut
        r = r[..., np.newaxis]
        rvec = rvec
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        grad = -rcut * a * c * rvec
        temp = 2 * (y - 1) / (rcut * r)
        temp -= a / r ** 2
        temp -= 2 * a * a * c * gamma * (1 + gamma * b)
        lap = -rcut * c * (a + rvec ** 2 * temp)
        grad[mask] = 0
        lap[mask] = 0
        return grad, lap

    def pgradient(self, rvec, r):
        """ Returns gradient of self.value with respect to all parameters
        Parameters:
          rvec: (nconf,...,3) vector
        Returns:
          paramderivs: dictionary {'rcut':d/drcut,'gamma':d/dgamma}
        """
        func = {"rcut": np.zeros(r.shape), "gamma": np.zeros(r.shape)}
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r <= rcut
        r = r[mask]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = a / (1 + gamma * b) ** 2 / (rcut * r)
        val = -b / (1 + gamma * b) + 1 / (3 + gamma)

        func["rcut"][mask] = y * a / (1 + gamma * b) ** 2 + val
        func["gamma"][mask] = ((b / (1 + gamma * b)) ** 2 - 1 / (3 + gamma) ** 2) * rcut

        return func


def test_func3d_gradient(bf, delta=1e-5):
    rvec = np.random.randn(150, 5, 10, 3)  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    numeric = np.zeros(rvec.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        pos[..., d] -= 2 * delta
        minuval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        numeric[..., d] = (plusval - minuval) / (2 * delta)
    meanerror = np.mean(np.abs(grad - numeric))
    normerror = np.linalg.norm(grad - numeric)
    return (meanerror, normerror)


def test_func3d_laplacian(bf, delta=1e-5):
    rvec = np.random.randn(150, 5, 10, 3)  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    lap = bf.laplacian(rvec, r)
    numeric = np.zeros(rvec.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        r = np.linalg.norm(pos, axis=-1)
        plusval = bf.gradient(pos, r)[..., d]
        pos[..., d] -= 2 * delta
        r = np.linalg.norm(pos, axis=-1)
        minuval = bf.gradient(pos, r)[..., d]
        numeric[..., d] = (plusval - minuval) / (2 * delta)
    meanerror = np.mean(np.abs(lap - numeric))
    normerror = np.linalg.norm(lap - numeric)
    return (meanerror, normerror)


def test_func3d_gradient_laplacian(bf):
    rvec = np.random.randn(150, 10, 3)
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    lap = bf.laplacian(rvec, r)
    andgrad, andlap = bf.gradient_laplacian(rvec, r)
    graderr = np.linalg.norm((grad - andgrad))
    laperr = np.linalg.norm((lap - andlap))
    return (graderr, laperr)


def test_func3d_pgradient(bf, delta=1e-5):
    rvec = np.random.randn(150, 10, 3)
    r = np.linalg.norm(rvec, axis=-1)
    pgrad = bf.pgradient(rvec, r)
    numeric = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    meanerror = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    normerror = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    for k in pgrad.keys():
        bf.parameters[k] += delta
        plusval = bf.value(rvec, r)
        bf.parameters[k] -= 2 * delta
        minuval = bf.value(rvec, r)
        bf.parameters[k] += delta
        numeric[k] = (plusval - minuval) / (2 * delta)
        meanerror[k] = np.mean(np.abs(pgrad[k] - numeric[k]))
        normerror[k] = np.linalg.norm(pgrad[k] - numeric[k])
        if meanerror[k] > 1e-5:
            print(k, "\n", pgrad[k] - numeric[k])
    return (meanerror, normerror)
