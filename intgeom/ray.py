from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import bisect

from intgeom.geod import solve_geod


@dataclass(frozen=True)
class Coords:
    Nph: int

    def to_canvas(self, x):
        return self.Nph * (x + 3) / 2

    def from_canvas(self, y):
        return y * 2 / self.Nph - 3

    def ph_from_canvas(self, ph):
        n = self.Nph
        xs = self.from_canvas(np.arange(n, 2 * n))
        # Interpolation with RegularGridInterpolator gives array-axis
        # order, that is, rows are indexed before columns. This is
        # different from the geometric xy-order. For this reason, we
        # pass transposed, flipped phantom to the interpolator.
        return RegularGridInterpolator(
            (xs, xs),
            np.fliplr(ph.T),
            method="cubic",
            bounds_error=False,
            fill_value=0,
        )


def sources(th=0, Nrays=10):
    rmat = np.array(
        [[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]]
    )
    return rmat @ np.stack(
        [np.full(Nrays, -1), np.linspace(-1, 1, Nrays)]
    ), rmat @ [1, 0]


def dist_to_src_line(p, th=0):
    _, src_dir = sources(th=th, Nrays=2)
    return src_dir @ p + 1


def exit_time(sol, th=0):
    # We assume that the goedesic given by the solver has exited
    def f(t):
        return dist_to_src_line(sol.sol(t)[0:2], th=th) - 2

    return bisect(f, sol.t[0], sol.t[-1])


def project(ph, Gammas, Nrays=20, th=0):
    out = np.zeros(Nrays)
    coords = Coords(Nph=ph.shape[0])
    ph_interp = coords.ph_from_canvas(ph)
    src_pts, src_dir = sources(th=th, Nrays=Nrays)
    for i in range(Nrays):
        src_pt = src_pts[:, i]
        sol = solve_geod(src_pt, src_dir, Gammas)
        t = exit_time(sol, th=th)

        def f(t, sol=sol):
            u = sol.sol(t)
            return float(ph_interp((u[0], u[1])))

        ts = np.linspace(0, t)
        fts = np.array([f(t) for t in ts])
        out[i] = simpson(fts, x=ts)
    return out
