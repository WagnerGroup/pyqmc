import numpy as np
import pyqmc.gpu as gpu

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


@gpu.fuse()
def polypadevalue(z, beta):
    p = z * z * (6 - 8 * z + 3 * z * z)
    return (1 - p) / (1 + beta * p)


@gpu.fuse()
def polypadegradvalue(r, beta, rcut):
    z = r / rcut
    p = z * z * (6 - 8 * z + 3 * z * z)
    dpdz = 12 * z * (z * z - 2 * z + 1)
    dbdp = -(1 + beta) / (1 + beta * p) ** 2
    dzdx_rvec = 1 / (r * rcut)
    grad_rvec = dbdp * dpdz * dzdx_rvec
    value = (1 - p) / (1 + beta * p)
    return grad_rvec, value


class PolyPadeFunction:
    r"""
    :math:`b(r) = \frac{1-p(z)}{1+\beta p(z)}`
    :math:`z = r/r_{\rm cut}`
    where
    :math:`p(z) = 6z^2 - 8z^3 + 3z^4`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, beta, rcut):
        self.parameters = {
            "beta": gpu.cp.asarray(beta),
            "rcut": gpu.cp.asarray(rcut),
        }

    def value(self, rvec, r):
        """Returns function (1-p(r/rcut))/(1+beta*p(r/rcut))

        :parameter rvec: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: function value
        :rtype: (nconfig,...) array
        """
        mask = r < self.parameters["rcut"]
        z = r[mask] / self.parameters["rcut"]
        func = gpu.cp.zeros(r.shape)
        func[mask] = polypadevalue(z, self.parameters["beta"])
        return func

    def gradient_value(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and value
        :rtype: tuple of (nconfig,...,3) arrays
        """
        value = gpu.cp.zeros(r.shape)
        grad = gpu.cp.zeros(rvec.shape)
        mask = r < self.parameters["rcut"]
        grad_rvec, value[mask] = polypadegradvalue(
            r[mask],
            self.parameters["beta"],
            self.parameters["rcut"],
        )
        grad[mask] = np.einsum("ij,i->ij", rvec[mask], grad_rvec)
        return grad, value

    def gradient(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient
        :rtype: (nconfig,...,3) array
        """
        grad = gpu.cp.zeros(rvec.shape)
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
        grad = gpu.cp.zeros(rvec.shape)
        lap = gpu.cp.zeros(rvec.shape)
        mask = r < self.parameters["rcut"]
        r = r[..., np.newaxis]
        r = r[mask]
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
        mask = r >= self.parameters["rcut"]
        z = r / self.parameters["rcut"]
        beta = self.parameters["beta"]

        p = z * z * (6 - 8 * z + 3 * z * z)
        dbdp = -(1 + beta) / (1 + beta * p) ** 2
        dpdz = 12 * z * (z * z - 2 * z + 1)
        derivrcut = dbdp * dpdz * (-z / self.parameters["rcut"])
        derivbeta = -p * (1 - p) / (1 + beta * p) ** 2
        derivrcut[mask] = 0.0
        derivbeta[mask] = 0.0
        pderiv = {"rcut": derivrcut, "beta": derivbeta}
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
        func = gpu.cp.zeros(r.shape)
        func[mask] = -p / (1 + gamma * p) + 1 / (3 + gamma)
        return func * self.parameters["rcut"]

    def gradient(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient
        :rtype: (nconfig,...,3) array
        """
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        r = r[mask][..., np.newaxis]
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * r)

        grad = gpu.cp.zeros(rvec.shape)
        grad[mask] = -rvec[mask] * a * c * rcut
        return grad

    def gradient_value(self, rvec, r):
        """
        :parameter rvec: (nconfig,...,3) 
        :parameter r: (nconfig,...) 
        :returns: gradient and value
        :rtype: tuple of (nconfig,...,3) arrays
        """
        grad = gpu.cp.zeros(rvec.shape)
        value = gpu.cp.zeros(r.shape)
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
        lap = gpu.cp.zeros(rvec.shape)
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
        grad = gpu.cp.zeros(rvec.shape)
        lap = gpu.cp.zeros(rvec.shape)
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
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r > rcut
        r = r
        y = r / rcut

        a = 1 - 2 * y + y * y
        b = y - y * y + y * y * y / 3
        c = a / (1 + gamma * b) ** 2 / (rcut * r)
        val = -b / (1 + gamma * b) + 1 / (3 + gamma)

        dfdrcut = y * a / (1 + gamma * b) ** 2 + val
        dfdgamma = ((b / (1 + gamma * b)) ** 2 - 1 / (3 + gamma) ** 2) * rcut
        dfdrcut[mask] = 0.0
        dfdgamma[mask] = 0.0
        func = {"rcut": dfdrcut, "gamma": dfdgamma}

        return func


class LPQHI:
    r"""A locally piecewise-quintic Hermite interpolant as defined in Natoli and Ceperley, J. Comp. Phys. 117, 171-178, (1995).

    :math:`h_{i\alpha}(r) = (\Delta)^\alpha \sum_{n=0}^5 S_{\alpha n} \left(\frac{r-r_i}{\Delta}\right)^n, \quad r_i < r \le r_{i+1}`
    :math:`h_{i\alpha}(r) = (-\Delta)^\alpha \sum_{n=0}^5 S_{\alpha n} \left(\frac{r_i-r}{\Delta}\right)^n, \quad r_{i-1} < r \le r_{i}`

    .. math: h_\alpha(r) = \sum_n t_n h_{n\alpha}(r)

    parameter :math:`t_{i\alpha}` is an array of shape ``(n+1, 3)``, where the last row should be zeros for a smooth cutoff
    parameter :math:`r_{\rm cut}` is the cutoff radius
    """

    def __init__(self, t, rcut):
        self.parameters = {"t": gpu.cp.asarray(t), "rcut": gpu.cp.asarray(rcut)}
        self.S_matrix = gpu.cp.array(
            [
                [1, 0, 0.0, -10.0, 15.0, -6.0],  #  0
                [0, 1, 0.0, -6.0, 8.0, -3.0],  #  1
                [0, 0, 0.5, -1.5, 1.5, -0.5],  #  2
            ]
        )
        deriv = gpu.cp.diag([1, 2, 3, 4, 5], k=-1)
        self.S_deriv1 = gpu.cp.dot(self.S_matrix, deriv)
        self.S_deriv2 = gpu.cp.dot(self.S_deriv1, deriv)

    @classmethod
    def initialize_random(self, nknots, rcut=7.5):
        t = np.random.random((nknots + 1, 3)) * 4 - 2
        t[-1] = 0
        return LPQHI(t, rcut)

    def value(self, rvec, r):
        mask = r < self.parameters["rcut"]
        r_ = r[mask]
        deltas = _lpqhi_get_deltas(self.parameters)

        m, dr_n = _lpqhi_pack_r(r_, deltas[0, 1])
        h_pm = gpu.cp.einsum("jik,lj,il->ikl", dr_n, self.S_matrix, deltas)
        t_pm = _lpqhi_select_t_pm(self.parameters["t"], m)

        f = gpu.cp.zeros(r.shape)
        f[mask] = gpu.cp.einsum("ikl,ikl->k", t_pm, h_pm)
        return f

    def gradient(self, rvec, r):
        mask = r < self.parameters["rcut"]
        r_ = r[mask]
        rvec_ = rvec[mask]
        deltas = _lpqhi_get_deltas(self.parameters)

        m, dr_n = _lpqhi_pack_r(r_, deltas[0, 1])
        sign = gpu.cp.array([1, -1])
        h_pm1 = gpu.cp.einsum("jik,lj,il,i->ikl", dr_n, self.S_deriv1, deltas / deltas[0, 1], sign)
        t_pm = _lpqhi_select_t_pm(self.parameters["t"], m)

        grad = gpu.cp.zeros(rvec.shape)
        grad[mask] = gpu.cp.einsum(
            "ikl,ikl,km->km", t_pm, h_pm1, rvec_ / r_[:, np.newaxis]
        )
        return grad

    def gradient_value(self, rvec, r):
        mask = r < self.parameters["rcut"]
        r_ = r[mask]
        rvec_ = rvec[mask]
        deltas = _lpqhi_get_deltas(self.parameters)

        m, dr_n = _lpqhi_pack_r(r_, deltas[0, 1])
        sign = gpu.cp.array([1, -1])
        h_pm = gpu.cp.einsum("jik,lj,il->ikl", dr_n, self.S_matrix, deltas)
        h_pm1 = gpu.cp.einsum("jik,lj,il,i->ikl", dr_n, self.S_deriv1, deltas / deltas[0, 1], sign)
        t_pm = _lpqhi_select_t_pm(self.parameters["t"], m)

        val = gpu.cp.zeros(r.shape)
        val[mask] = gpu.cp.einsum("ikl,ikl->k", t_pm, h_pm)
        grad = gpu.cp.zeros(rvec.shape)
        grad[mask] = gpu.cp.einsum(
            "ikl,ikl,km->km", t_pm, h_pm1, rvec_ / r_[:, np.newaxis]
        )
        return grad, val

    def laplacian(self, rvec, r):
        return self.gradient_laplacian(rvec, r)[1]

    def gradient_laplacian(self, rvec, r):
        mask = r < self.parameters["rcut"]
        r_ = r[mask]
        rvec_ = rvec[mask]
        deltas = _lpqhi_get_deltas(self.parameters)

        m, dr_n = _lpqhi_pack_r(r_, deltas[0, 1])
        sign = gpu.cp.array([1, -1])
        h_pm1 = gpu.cp.einsum("jik,lj,il,i->ikl", dr_n, self.S_deriv1, deltas / deltas[0, 1], sign)
        h_pm2 = gpu.cp.einsum("jik,lj,il->ikl", dr_n, self.S_deriv2, deltas / deltas[0, 1] ** 2)
        t_pm = _lpqhi_select_t_pm(self.parameters["t"], m)

        dfdr = gpu.cp.einsum("ikl,ikl->k", t_pm, h_pm1)[:, np.newaxis]
        d2fdr2 = gpu.cp.einsum("ikl,ikl->k", t_pm, h_pm2)[:, np.newaxis]
        drdx = rvec_ / r_[:, np.newaxis]
        d2rdx2 = (1 - drdx ** 2) / r_[:, np.newaxis]

        grad = gpu.cp.zeros(rvec.shape)
        lap = gpu.cp.zeros(rvec.shape)
        grad[mask] = dfdr * drdx
        lap[mask] = d2fdr2 * drdx ** 2 + dfdr * d2rdx2
        return grad, lap

    def pgradient(self, rvec, r):
        mask = r < self.parameters["rcut"]
        r_ = r[mask]
        rvec_ = rvec[mask]
        deltas = _lpqhi_get_deltas(self.parameters)

        m, dr_n = _lpqhi_pack_r(r_, deltas[0, 1])
        sign = gpu.cp.array([1, -1])
        h_pm = gpu.cp.einsum("jik,lj,il->ikl", dr_n, self.S_matrix, deltas)

        pgrad_mask = np.zeros((*r_.shape, *self.parameters["t"].shape))
        m = m.astype(int)
        pgrad_mask[np.arange(len(m)), m] = h_pm[0]
        pgrad_mask[np.arange(len(m)), m + 1] = h_pm[1]

        pgrad = np.zeros((*r.shape, *self.parameters["t"].shape))
        pgrad[mask] = pgrad_mask
        return {"t": pgrad}


def _lpqhi_select_t_pm(t, m):
    m = m.astype(int)
    t_pm = gpu.cp.zeros((2, len(m), 3))
    t_pm[0] = t[m]
    t_pm[1] = t[m + 1]
    return t_pm


def _lpqhi_get_deltas(parameters):
    delta = parameters["rcut"] / (parameters["t"].shape[0] - 1)
    deltas = gpu.cp.array([[1, delta, delta ** 2], [1, -delta, delta ** 2]])
    return deltas


def _lpqhi_pack_r(r_, delta):
    m, dr = gpu.cp.divmod(r_ / delta, 1.0)
    dr_pm = gpu.cp.stack([dr, 1 - dr], axis=0)
    dr_n = gpu.cp.stack([dr_pm ** n for n in range(6)], axis=0)
    return m, dr_n


def test_func3d_gradient(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 5, 10, 3))  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    numeric = gpu.cp.zeros(rvec.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        pos[..., d] -= 2 * delta
        minuval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        numeric[..., d] = (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(grad - numeric))
    return gpu.asnumpy(maxerror)


def test_func3d_laplacian(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 5, 10, 3))  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    lap = bf.laplacian(rvec, r)
    numeric = gpu.cp.zeros(rvec.shape)
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
    return gpu.asnumpy(maxerror)


def test_func3d_gradient_laplacian(bf):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    lap = bf.laplacian(rvec, r)
    andgrad, andlap = bf.gradient_laplacian(rvec, r)
    graderr = np.amax(np.abs(grad - andgrad))
    laperr = np.amax(np.abs(lap - andlap))
    return {"grad": graderr, "lap": laperr}


def test_func3d_gradient_value(bf):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    val = bf.value(rvec, r)
    andgrad, andval = bf.gradient_value(rvec, r)
    graderr = np.linalg.norm((grad - andgrad))
    valerr = np.linalg.norm((val - andval))
    return {"grad": graderr, "val": valerr}


def test_func3d_pgradient(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    pgrad = bf.pgradient(rvec, r)
    numeric = {
        k: gpu.cp.zeros((r.size, np.size(v))) for k, v in bf.parameters.items()
    }
    error = {}
    for k in pgrad.keys():
        flt = np.reshape(bf.parameters[k], -1)
        shape = np.shape(bf.parameters[k])
        for i, c in enumerate(flt):
            flt[i] += delta
            bf.parameters[k] = flt.reshape(shape)
            plusval = bf.value(rvec, r)
            flt[i] -= 2 * delta
            bf.parameters[k] = flt.reshape(shape)
            minuval = bf.value(rvec, r)
            flt[i] += delta
            bf.parameters[k] = flt.reshape(shape)
            numeric[k][:, i] = (plusval - minuval).ravel() / (2 * delta)
        pgerr = np.abs(pgrad[k].reshape((-1, len(flt))) - numeric[k])
        error[k] = gpu.asnumpy(np.amax(pgerr))
    if len(error) == 0:
        return (0, 0)
    return error

