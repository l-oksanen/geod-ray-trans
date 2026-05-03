import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp


def Gammas(c, symbols=None):
    """
    Return Christoffel symbols for conformally Euclidean metric.

    Given a conformal factor c as a symbolic expression with free
    symbols given by coords, the Christoffel symbols \Gamma^i_kl are
    returned as a function taking values in arrays indexed by i, k, l.
    """

    if symbols is None:
        symbols = sorted(c.free_symbols, key=str)
    n = len(symbols)
    g = c ** (-2) * sp.eye(n)
    g_inv = c**2 * sp.eye(n)
    G = sp.MutableDenseNDimArray.zeros(n, n, n)
    for i, k, l, m in [
        (i, k, l, m)
        for i in range(n)
        for k in range(n)
        for l in range(n)
        for m in range(n)
    ]:
        G[i, k, l] += (
            sp.Rational(1, 2)
            * g_inv[i, m]
            * (
                sp.diff(g[m, k], symbols[l])
                + sp.diff(g[m, l], symbols[k])
                - sp.diff(g[k, l], symbols[m])
            )
        )
    return sp.lambdify(symbols, sp.simplify(G), cse=True)


def G(u, Gammas):
    """
    Return geodesic vector field at point u on the tangent bundle.

    The Christoffel symbols Gammas should be as returned by the
    function Gammas.
    """
    out = np.array([u[2], u[3], 0, 0])
    for k in [0, 1]:
        for l in [0, 1]:
            out[2] -= (
                Gammas(u[0], u[1])[0, k, l] * u[k + 2] * u[l + 2]
            )
            out[3] -= (
                Gammas(u[0], u[1])[1, k, l] * u[k + 2] * u[l + 2]
            )
    return out


def solve_geod(src_pt, src_dir, Gammas, t_max=5.5):
    """
    Return geodesic with initial point src_pt and direction src_dir.

    The Christoffel symbols Gammas should be as returned by the
    function Gammas. The geodesic is solved on (0, t_max) and it is
    returned in the form given by solve_ivp of SciPy.
    """
    u0 = np.concatenate([src_pt, src_dir])

    def f(t, u):
        return G(u, Gammas)

    return solve_ivp(
        f,
        (0, t_max),
        u0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
