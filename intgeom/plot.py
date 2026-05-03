from typing import NamedTuple

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb

from intgeom.ray import exit_time, sources

default_palette: dict = {
    "purple": np.array([186, 0, 255]) / 255,
    "yellow": np.array([246, 255, 0]) / 255,
    "detcolor": np.array([70, 173, 211]) / 255,
    "Xraycolor": np.array([236, 86, 47]) / 255,
    "Xray_darker_color": np.array([151, 55, 30]) / 255,
    "matrix green": "#00ff41",
}


class PlotParams(NamedTuple):
    """
    Parameters controlling plotting.

    msize         : source-dot marker size
    detlength     : detector line length
    detwidth      : detector line width
    raywidth      : X-ray line width
    datalinewidth : projection-data line width
    ccontourwidth : contour line width for wave speed
    projamp       : amplitude of projection data in plot units
    projoff       : offset of projection data from detector
    gammacorr     : gamma correction for the displayed phantom
    palette       : color palette
    """

    msize: int = 6
    detlength: float = 3
    detwidth: float = 1.2
    raywidth: float = 1
    datalinewidth: float = 2
    ccontourwidth: float = 1
    projamp: float = 1
    projoff: float = 0.15
    gammacorr: float = 0.5
    palette: dict = default_palette


default_pp = PlotParams()


def plot_palette(palette=default_palette):
    fig, axes = plt.subplots(1, len(palette))
    scaling = 1.2
    fig.set_size_inches(scaling * len(palette), scaling)
    for ax, (name, value) in zip(axes, palette.items()):
        plt.sca(ax)
        if isinstance(value, str):
            rgb = to_rgb(value)
        else:
            rgb = value
        plt.imshow(np.tile(rgb, (1, 1, 1)))
        plt.text(
            0.5,
            -0.15,
            name,
            transform=ax.transAxes,
            rotation=-45,
            ha="left",
            va="top",
            clip_on=False,
            rotation_mode="anchor",
        )
        plt.axis("off")


def plot_init(coords, color=(0, 0, 0)):
    n = 3 * coords.Nph
    # By default, imshow gives inverted orientation for y-axis, we
    # set usual, non-inverted orientation for the axes using
    # origin.
    plt.imshow(np.full((n, n, 3), color), extent=(0, n, 0, n))
    plt.gca().set_autoscale_on(False)


def plot_ph(coords, ph, pp=default_pp):
    n = coords.Nph
    plt.gca().imshow(
        np.power(ph, pp.gammacorr),
        extent=(n, 2 * n, n, 2 * n),
        cmap="gray",
    )


def plot_c(coords, c_sym, c_bound=None, symbols=None, pp=default_pp):
    if symbols is None:
        symbols = sorted(c_sym.free_symbols, key=str)
    c = sp.lambdify(symbols, c_sym)
    xs = np.linspace(-3, 3, 3 * coords.Nph)
    Xs, Ys = np.meshgrid(xs, xs)
    Cs = np.log(c(Xs, Ys))
    if c_bound is None:
        rel_margin = 0.1
        c_bound = (1 + rel_margin) * np.max(np.abs(Cs))
    if c_bound != 0:
        levels = np.linspace(-c_bound, c_bound, 13)
        plt.contourf(
            coords.to_canvas(Xs),
            coords.to_canvas(Ys),
            Cs,
            cmap="berlin",
            levels=levels,
            norm=TwoSlopeNorm(vmin=-c_bound, vcenter=0, vmax=c_bound),
        )
        plt.contour(
            coords.to_canvas(Xs),
            coords.to_canvas(Ys),
            Cs,
            cmap="berlin",
            levels=levels,
            norm=TwoSlopeNorm(vmin=-c_bound, vcenter=0, vmax=c_bound),
            linewidths=pp.ccontourwidth,
        )


def plot_sources(coords, src_pts, pp=default_pp):
    plt.plot(
        *coords.to_canvas(src_pts),
        ".",
        color=pp.palette["Xray_darker_color"],
        markersize=pp.msize,
    )


def plot_pr(coords, pr, pr_max, th=0, pp=default_pp):
    src_pts, src_dir = sources(th=th, Nrays=2)

    p0 = src_pts[:, 0]
    p1 = src_pts[:, 1]

    amp = pp.projoff + pr * pp.projamp / pr_max
    ss = np.linspace(0, 1, len(amp))
    ps = np.array(
        [
            s * p1 + (1 - s) * p0 - a * src_dir
            for (s, a) in zip(ss, amp)
        ]
    )
    plt.plot(
        *coords.to_canvas(ps.T),
        "-",
        color=pp.palette["yellow"],
        linewidth=pp.datalinewidth,
    )


def plot_detector(coords, th=0, pp=default_pp):
    src_pts, src_dir = sources(th=th, Nrays=2)
    p0 = src_pts[:, 0] + 2 * src_dir
    p1 = src_pts[:, 1] + 2 * src_dir
    p = (p1 + p0) / 2
    v = p1 - p0
    v = v / np.linalg.norm(v)
    p0 = p - v * pp.detlength / 2
    p1 = p + v * pp.detlength / 2

    plt.plot(
        *coords.to_canvas(np.stack([p0, p1]).T),
        "-",
        color=pp.palette["detcolor"],
        linewidth=pp.detwidth,
    )


def plot_geod(coords, sol, th=0, pp=default_pp):
    t = exit_time(sol, th=th)
    ts = np.linspace(0, t)
    gamma = sol.sol(ts)[0:2, :]

    plt.plot(
        *coords.to_canvas(gamma),
        "-",
        color=pp.palette["Xraycolor"],
        linewidth=pp.raywidth,
    )
