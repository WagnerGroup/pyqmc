import numpy as np

""" 
Collection of 3d function objects. Each has a dictionary parameters, which corresponds
to any variational parameters the funtion has.

They should implement the following functions, all of which take input values (x, r).
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

    :math:`\exp(-\alpha r^2)`
    where :math:`\alpha` can be accessed through parameters['exponent']

    """

    def __init__(self, exponent):
        self.parameters = {"exponent": exponent}

    def value(self, x, r):
        """Returns function exp(-exponent*r^2).

        :parameter x: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: function value
        :rtype: (nconfig,...) array
        """
        return np.exp(-self.parameters["exponent"] * r * r)

    def gradient(self, x, r):
        """Returns gradient of function.

        :parameter x: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: gradient
        :rtype: (nconfig,...,3)
        """
        v = self.value(x, r)
        return -2 * self.parameters["exponent"] * x * v[..., np.newaxis]

    def gradient_value(self, x, r):
        """
        :parameter x: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and value
        :rtype: tuple of (nconfig,...,3) arrays
        """
        v = self.value(x, r)
        g = -2 * self.parameters["exponent"] * x * v[..., np.newaxis]
        return g, v

    def laplacian(self, x, r):
        """Returns laplacian of function.

        :parameter x: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: laplacian (components of laplacian d^2/dx_i^2 separately)
        :rtype: (nconfig,...,3)
        """
        v = self.value(x, r)
        alpha = self.parameters["exponent"]
        return (4 * alpha * alpha * x * x - 2 * alpha) * v[..., np.newaxis]

    def gradient_laplacian(self, x, r):
        """Returns gradient and laplacian of function.

        :parameter x: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: gradient and laplacian
        :rtype: tuple of two (nconfig,...,3) arrays (components of laplacian d^2/dx_i^2 separately)
        """
        v = self.value(x, r)[..., np.newaxis]
        alpha = self.parameters["exponent"]
        grad = -2 * alpha * x * v
        lap = (4 * alpha * alpha * x * x - 2 * alpha) * v
        return grad, lap

    def pgradient(self, x, r):
        """Returns parameters gradient.

        :parameter x: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: parameter gradient {'exponent': d/dexponent}
        :rtype: dictionary
        """
        r2 = r * r
        return {"exponent": -r2 * np.exp(-self.parameters["exponent"] * r2)}


class PadeFunction:
    r"""
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    alpha_k = alpha/2^k, k starting at 0

    :math:`a_k(r) = \left( \frac{\alpha_k r}{1 + \alpha_k r} \right)^2`
    where
    :math:`\alpha_k = \frac{\alpha}{2^k}`, :math:`k` starting at 0
    """

    def __init__(self, alphak):
        self.parameters = {"alphak": alphak}

    def value(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: function value
        :rtype: (nconfig,...) array
        """
        a = self.parameters["alphak"] * r
        return (a / (1 + a)) ** 2

    def gradient(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient
        :rtype: (nconfig,...,3) array
        """
        a = self.parameters["alphak"] * r[..., np.newaxis]
        return 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3 * rvec

    def gradient_value(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and value
        :rtype: tuple of (nconfig,...,3) arrays
        """
        a = self.parameters["alphak"] * r
        value = (a / (1 + a)) ** 2
        a = a[..., np.newaxis]
        grad = 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3 * rvec
        return grad, value

    def laplacian(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: laplacian (returns components of laplacian d^2/dx_i^2 separately)
        :rtype: (nconfig,...,3) array
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

        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and laplacian (returns components of laplacian d^2/dx_i^2 separately)
        :rtype: tuple of (nconfig,...,3) arrays
        """
        a = self.parameters["alphak"] * r[..., np.newaxis]
        temp = 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3
        grad = temp * rvec
        lap = temp * (1 - 3 * a / (1 + a) * (rvec / r[..., np.newaxis]) ** 2)
        return grad, lap

    def pgradient(self, rvec, r):
        """Returns gradient with respect to parameter alphak

        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: parameter gradient  {'alphak': akderiv}
        :rtype: dictionary
        """
        a = self.parameters["alphak"] * r
        akderiv = 2 * a / (1 + a) ** 3 * r
        return {"alphak": akderiv}


class PolyPadeFunction:
    r"""
    :math:`b(r) = \frac{1-p(z)}{1+\beta p(z)}`
    :math:`z = r/r_{\rm cut}`
    where
    :math:`p(z) = 6z^2 - 8z^3 + 3z^4`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, beta, rcut):
        self.parameters = {"beta": beta, "rcut": rcut}

    def value(self, rvec, r):
        """Returns function (1-p(r/rcut))/(1+beta*p(r/rcut))

        :parameter rvec: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: function value
        :rtype: (nconfig,...) array
        """
        mask = r < self.parameters["rcut"]
        z = r[mask] / self.parameters["rcut"]
        p = z * z * (6 - 8 * z + 3 * z * z)
        func = np.zeros(r.shape)
        func[mask] = (1 - p) / (1 + self.parameters["beta"] * p)
        return func

    def gradient_value(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and value
        :rtype: tuple of (nconfig,...,3) arrays
        """
        value = np.zeros(r.shape)
        grad = np.zeros(rvec.shape)
        mask = r < self.parameters["rcut"]
        r = r[mask][..., np.newaxis]
        rvec = rvec[mask]
        z = r / self.parameters["rcut"]
        p = z * z * (6 - 8 * z + 3 * z * z)
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dbdp = -(1 + self.parameters["beta"]) / (1 + self.parameters["beta"] * p) ** 2
        dzdx = rvec / (r * self.parameters["rcut"])
        grad[mask] = dbdp * dpdz * dzdx
        p = p[..., 0]
        value[mask] = (1 - p) / (1 + self.parameters["beta"] * p)
        return grad, value

    def gradient(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient
        :rtype: (nconfig,...,3) array
        """
        grad = np.zeros(rvec.shape)
        mask = r < self.parameters["rcut"]
        r = r[mask][..., np.newaxis]
        rvec = rvec[mask]
        z = r / self.parameters["rcut"]
        p = z * z * (6 - 8 * z + 3 * z * z)
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dbdp = -(1 + self.parameters["beta"]) / (1 + self.parameters["beta"] * p) ** 2
        dzdx = rvec / (r * self.parameters["rcut"])
        grad[mask] = dbdp * dpdz * dzdx
        return grad

    def laplacian(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: laplacian
              returns components of laplacian d^2/dx_i^2 separately
        :rtype: (nconfig,...,3) array
        """
        return self.gradient_laplacian(rvec, r)[1]

    def gradient_laplacian(self, rvec, r):
        """Returns gradient and laplacian of function.

        :parameter x: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: gradient and laplacian
        :rtype: tuple of two (nconfig,...,3) arrays (components of laplacian d^2/dx_i^2 separately)
        """
        grad = np.zeros(rvec.shape)
        lap = np.zeros(rvec.shape)
        mask = r < self.parameters["rcut"]
        r = r[mask, np.newaxis]
        rvec = rvec[mask]
        z = r / self.parameters["rcut"]
        beta = self.parameters["beta"]

        p = z * z * (6 - 8 * z + 3 * z * z)
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dbdp = -(1 + beta) / (1 + beta * p) ** 2
        dzdx = rvec / (r * self.parameters["rcut"])
        gradmask = dbdp * dpdz * dzdx
        d2pdz2_over_dpdz = (3 * z - 1) / (z * (z - 1))
        d2bdp2_over_dbdp = -2 * beta / (1 + beta * p)
        d2zdx2 = (1 - (rvec / r) ** 2) / (r * self.parameters["rcut"])
        grad[mask] = gradmask
        lap[mask] += dbdp * dpdz * d2zdx2 + (
            gradmask * (d2bdp2_over_dbdp * dpdz * dzdx + d2pdz2_over_dpdz * dzdx)
        )
        return grad, lap

    def pgradient(self, rvec, r):
        """Returns gradient of self.value with respect to all parameters

        :parameter rvec: (nconf,...,3)
        :parameter r: (nconf,...)

        :return paramderivs: dictionary {'rcut':d/drcut,'beta':d/dbeta}
        """
        pderiv = {"rcut": np.zeros(r.shape), "beta": np.zeros(r.shape)}
        mask = r < self.parameters["rcut"]
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
    :math:`b(r) = -\frac{p(r/r_{\rm cut})}{1+\gamma*p(r/r_{\rm cut})} + \frac{1}{3+\gamma}`
    where
    :math:`p(y) = y - y^2 + y^3/3`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, gamma, rcut):
        self.parameters = {"gamma": gamma, "rcut": rcut}

    def value(self, rvec, r):
        """Returns function  p(r/rcut)/(1+gamma*p(r/rcut))

        :parameter rvec: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: function value
        :rtype: (nconfig,...) array
        """
        mask = r < self.parameters["rcut"]
        y = r[mask] / self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        p = y - y * y + y * y * y / 3
        func = np.zeros(r.shape)
        func[mask] = -p / (1 + gamma * p) + 1 / (3 + gamma)
        return func * self.parameters["rcut"]

    def gradient(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient
        :rtype: (nconfig,...,3) array
        """
        grad = np.zeros(rvec.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        r = r[mask][..., np.newaxis]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        grad[mask] = -rvec[mask] * a * c * rcut
        return grad

    def gradient_value(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and value
        :rtype: tuple of (nconfig,...,3) arrays
        """
        grad = np.zeros(rvec.shape)
        value = np.zeros(r.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        r = r[mask][..., np.newaxis]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        grad[mask] = -rvec[mask] * a * c * rcut
        b = b[..., 0]
        value[mask] = -b / (1 + gamma * b) + 1 / (3 + gamma)
        return grad, value * rcut

    def laplacian(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: laplacian (returns components of laplacian d^2/dx_i^2 separately)
        :rtype: (nconfig,...,3) array
        """
        lap = np.zeros(rvec.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        r = r[mask][..., np.newaxis]
        rvec = rvec[mask]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        temp = 2 * (y - 1) / (rcut * r)
        temp -= a / r ** 2
        temp -= 2 * a * a * c * gamma * (1 + gamma * b)
        lap[mask] = -rcut * c * (a + rvec ** 2 * temp)
        return lap

    def gradient_laplacian(self, rvec, r):
        """Returns gradient and laplacian of function.

        :parameter rvec: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: gradient and laplacian
        :rtype: tuple of two (nconfig,...,3) arrays (components of laplacian d^2/dx_i^2 separately)
        """
        grad = np.zeros(rvec.shape)
        lap = np.zeros(rvec.shape)
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        r = r[mask][..., np.newaxis]
        rvec = rvec[mask]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        grad[mask] = -rcut * a * c * rvec
        temp = 2 * (y - 1) / (rcut * r)
        temp -= a / r ** 2
        temp -= 2 * a * a * c * gamma * (1 + gamma * b)
        lap[mask] = -rcut * c * (a + rvec ** 2 * temp)
        return grad, lap

    def pgradient(self, rvec, r):
        """Returns gradient of self.value with respect to all parameters

        :parameter rvec: (nconf,...,3) 
        :parameter r: (nconfig,...) 
        :returns: parameter derivatives {'rcut':d/drcut,'gamma':d/dgamma}
        :rtype: dict
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
    maxerror = np.max(np.abs(grad - numeric))
    return maxerror


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
    maxerror = np.max(np.abs(lap - numeric))
    return maxerror


def test_func3d_gradient_laplacian(bf):
    rvec = np.random.randn(150, 10, 3)
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    lap = bf.laplacian(rvec, r)
    andgrad, andlap = bf.gradient_laplacian(rvec, r)
    graderr = np.amax(np.abs(grad - andgrad))
    laperr = np.amax(np.abs(lap - andlap))
    return {"grad": graderr, "lap": laperr}


def test_func3d_gradient_value(bf):
    rvec = np.random.randn(150, 10, 3)
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    val = bf.value(rvec, r)
    andgrad, andval = bf.gradient_value(rvec, r)
    graderr = np.linalg.norm((grad - andgrad))
    valerr = np.linalg.norm((val - andval))
    return {"grad": graderr, "val": valerr}


def test_func3d_pgradient(bf, delta=1e-5):
    rvec = np.random.randn(150, 10, 3)
    r = np.linalg.norm(rvec, axis=-1)
    pgrad = bf.pgradient(rvec, r)
    numeric = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    maxerror = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    normerror = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    for k in pgrad.keys():
        bf.parameters[k] += delta
        plusval = bf.value(rvec, r)
        bf.parameters[k] -= 2 * delta
        minuval = bf.value(rvec, r)
        bf.parameters[k] += delta
        numeric[k] = (plusval - minuval) / (2 * delta)
        maxerror[k] = np.max(np.abs(pgrad[k] - numeric[k]))
        if maxerror[k] > 1e-5:
            print(k, "\n", pgrad[k] - numeric[k])
    return maxerror
