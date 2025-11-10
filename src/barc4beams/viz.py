# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025 Synchrotron SOLEIL

"""
viz.py - plotting routines for beams and beamline layouts.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from itertools import cycle
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParamsDefault
from matplotlib.colors import Colormap
from scipy.stats import gaussian_kde, moment

from . import stats
from .propagation import _propagate_xy

Number = Union[int, float]
RangeT = Optional[Tuple[Optional[Number], Optional[Number]]]
BinsT  = Optional[Union[int, Tuple[int, int]]]
ModeT  = Union[Literal["scatter", "hist2d"], str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot() -> None:
    """Show all pending figures."""
    plt.show()

def plot_beam(
    df: pd.DataFrame,
    *,
    mode: str = "scatter",
    aspect_ratio: bool = True,
    color = 1,
    x_range = None,
    y_range = None,
    bins = None,
    bin_width = None,
    bin_method = 0,
    dpi: int = 100,
    path: Optional[str] = None,
    showXhist=True,
    showYhist=True,
    envelope=False,
    envelope_method="edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
    z_offset: float = 0.0
):
    """
    Plot the spatial footprint of a standardized beam (X vs Y), with optional marginals
    and moment-matched envelope overlays.

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized beam with columns: 'X','Y','dX','dY','lost_ray_flag' (0=alive).
        Units expected in meters (pos) and radians (angles). This function scales to µm/µrad.
    mode : {'scatter','histo2d', ...}, default 'scatter'
        Plot style. Aliases like 's'/'h' are accepted and normalized.
    aspect_ratio : bool, default True
        If True, main axes uses equal aspect.
    color : int or None, default 1
        Legacy color scheme index. 0/None → monochrome points; 1..4 → colormaps.
    x_range, y_range : (min, max) or None
        Data limits. If None/partial, auto-detected with a small padding.
    bins : int or (x_bins, y_bins) or None
        Histogram binning for the marginals and hist2d. Auto if None.
    bin_width : float or None
        If given, overrides bin count as ceil(range/bin_width).
    bin_method : int, default 0
        Auto-binning rule: 0=sqrt, 1=Sturges, 2=Rice, 3=Doane.
    dpi : int, default 300
    path : str or None
        If provided, the figure is saved.
    showXhist, showYhist : bool, default True
        Whether to show X/Y marginals.
    envelope : bool, default True
        Overlay envelope curve on the 1D marginals using moments from the data.
    envelope_method : {'edgeworth','pearson','maxent'}, default 'edgeworth'
        Reconstruction method passed to `stats.calc_envelope_from_moments`.
    apply_style : bool, default True
        Call `start_plotting(k)` before plotting.
    k : float, default 1.0
        Global style scale factor.

    Returns
    -------
    fig, (ax_image, ax_histx, ax_histy)
        The Matplotlib figure and axes.
    """

    if apply_style:
        start_plotting(k)

    x, y, xl, yl = _prep_beam_xy(df, kind="size", z_offset=z_offset)
    fig, axes = _common_xy_plot(
        x, y, xl, yl, _resolve_mode(mode), aspect_ratio, color,
        x_range, y_range, bins, bin_width, bin_method, dpi, path,
        showXhist, showYhist, envelope, envelope_method
    )
    if plot:
        plt.show()

    return fig, axes

def plot_divergence(
    df: pd.DataFrame,
    *,
    mode: str = "scatter",
    aspect_ratio: bool = False,
    color = 2,
    x_range = None,
    y_range = None,
    bins = None,
    bin_width = None,
    bin_method = 0,
    dpi: int = 100,
    path: Optional[str] = None,
    showXhist=True,
    showYhist=True,
    envelope=False,
    envelope_method="edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
    z_offset: float = 0.0
):
    """
    Plot the beam divergence (dX vs dY) in µrad with optional marginals and envelopes.
    (See `plot_beam` for parameter semantics.)

    Returns
    -------
    fig, (ax_image, ax_histx, ax_histy)
    """
    if apply_style:
        start_plotting(k)

    x, y, xl, yl = _prep_beam_xy(df, kind="div", z_offset=0)
    fig, axes = _common_xy_plot(
        x, y, xl, yl, _resolve_mode(mode), aspect_ratio, color,
        x_range, y_range, bins, bin_width, bin_method, dpi, path,
        showXhist, showYhist, envelope, envelope_method
    )
    if plot:
        plt.show()
    return fig, axes

def plot_phase_space(
    df: pd.DataFrame,
    *,
    direction: str = "both",
    mode: str = "scatter",
    aspect_ratio: bool = False,
    color = 3,
    x_range = None,
    y_range = None,
    bins = None,
    bin_width = None,
    bin_method = 0,
    dpi: int = 100,
    path: Optional[str] = None,
    showXhist=True,
    showYhist=True,
    envelope=False,
    envelope_method="edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
    z_offset: float = 0.0,
):
    """
    Plot phase space for one or both planes: (X vs dX) and/or (Y vs dY), in µm/µrad.

    Returns
    -------
    (fig_x, axes_x), (fig_y, axes_y)  if direction='both'
    or
    fig, (ax_image, ax_histx, ax_histy)
    """
    if apply_style:
        start_plotting(k)

    dnorm = str(direction).strip().lower()
    if dnorm not in {"x", "y", "both"}:
        import warnings
        warnings.warn(f"direction {direction!r} not recognized. Falling back to 'both'.")
        dnorm = "both"

    def _suffix(base: Optional[str], suf: str) -> Optional[str]:
        if not base:
            return None
        stem, ext = (base.rsplit(".", 1) + ["png"])[:2]
        return f"{stem}{suf}.{ext}"

    def _one(d: str, save_path: Optional[str]):
        x, y, xl, yl = _prep_beam_xy(df, kind="ps", direction=d, z_offset=z_offset)
        return _common_xy_plot(
            x, y, xl, yl, _resolve_mode(mode), aspect_ratio, color,
            x_range, y_range, bins, bin_width, bin_method, dpi, save_path,
            showXhist, showYhist, envelope, envelope_method
        )

    if dnorm == "both":
        fig_x, axes_x = _one("x", _suffix(path, "_x_dx"))
        fig_y, axes_y = _one("y", _suffix(path, "_y_dy"))
        if plot:
            plt.show()
        return (fig_x, axes_x), (fig_y, axes_y)

    fig, axes = _one(dnorm, path)
    if plot:
        plt.show()
    return fig, axes

def plot_caustic(
    caustic: dict,
    *,
    which: Literal["x", "y", "both"] = "both",
    aspect_ratio: bool = False,           # z vs X/Y typically very different scales
    color: Optional[int] = 5,
    z_range: RangeT = None,               # optional clamp of z-lims (display only)
    xy_range: RangeT = None,              # optional clamp of x/y (µm) display/bins
    bins: BinsT = None,                   # int or (None, int) → number of position bins
    bin_width: Optional[Number] = None,   # µm; overrides bins for position axis
    dpi: int = 100,
    path: Optional[str] = None,           # when which='both', adds _x/_y suffixes
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
    top_stat: Optional[str] = None        # 'fwhm' or 'std' or None
):
    """
    Plane-centered 2D histogram caustic (no other modes allowed).

    - Z is binned with one **contiguous** column per plane (no gaps).
    - X/Y use **fixed** bin width (if `bin_width`) or a fixed number of bins (`bins`).
    - Optional top panel shows FWHM or sigma (STD) vs z.

    Notes
    -----
    * Units: position on the map is in µm (input arrays in meters are scaled by 1e6).
    * Large beams are safe: uses numpy.histogram2d + pcolormesh (no KDE, no scatter).
    """
    if apply_style:
        start_plotting(k)

    z = np.asarray(caustic["optical_axis"], dtype=float)
    cm = caustic.get("caustic", {})
    Xmat = cm.get("X", None); Ymat = cm.get("Y", None)
    if which in ("x", "both") and Xmat is None:
        raise ValueError("X matrix not present in caustic['caustic']['X'].")
    if which in ("y", "both") and Ymat is None:
        raise ValueError("Y matrix not present in caustic['caustic']['Y'].")

    z_edges = _make_plane_centered_edges(z)

    def _apply_z_display_limits(ax):
        if z_range is not None and all(v is not None for v in z_range):
            ax.set_xlim(float(z_range[0]), float(z_range[1]))
        else:
            ax.set_xlim(z_edges[0], z_edges[-1])

    def _pos_edges(mat, rng):
        mat = np.asarray(mat, dtype=float)
        pos = mat.reshape(-1) * 1e6  # meters -> µm
        if rng is None or rng[0] is None or rng[1] is None:
            finite = pos[np.isfinite(pos)]
            lo = float(np.min(finite)) if finite.size else 0.0
            hi = float(np.max(finite)) if finite.size else 1.0
            if rng is not None:
                lo = rng[0] if rng[0] is not None else lo
                hi = rng[1] if rng[1] is not None else hi
        else:
            lo, hi = float(rng[0]), float(rng[1])

        nb = None
        if isinstance(bins, (tuple, list)) and len(bins) == 2 and isinstance(bins[1], int):
            nb = bins[1]
        elif isinstance(bins, int):
            nb = bins
        return _edges_from_span(lo, hi, bin_width=bin_width, bins=nb)

    def _plot_one(axis_key: str, mat):
        pos_edges = _pos_edges(mat, xy_range)
        P, N = np.asarray(mat).shape
        z_rep = np.repeat(z, N)
        pos_um = np.asarray(mat, dtype=float).reshape(P * N) * 1e6
        H, ze, xe = np.histogram2d(z_rep, pos_um, bins=[z_edges, pos_edges])

        if top_stat:
            fig = plt.figure(figsize=(10, 5.0), dpi=dpi)
            gs = fig.add_gridspec(
                nrows=2, ncols=2,
                width_ratios=[20, 1],    # main plot + colorbar
                height_ratios=[2, 4],    # top panel shorter than main
                hspace=0.05, wspace=0.05
            )
            axtop = fig.add_subplot(gs[0, 0])
            ax    = fig.add_subplot(gs[1, 0])
            cax   = fig.add_subplot(gs[:, 1])  # spans both rows, vertical colorbar
        else:
            fig, ax = plt.subplots(figsize=(10, 4.0), dpi=dpi)
            axtop = None
            cax = None

        mesh = ax.pcolormesh(ze, xe, H.T, shading='flat', cmap=_color_palette(color or 2))
        if aspect_ratio:
            ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r"$z$ [m]")
        ax.set_ylabel(r"$%s$ [$\mu$m]" % ("x" if axis_key == "x" else "y"))
        ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
        _apply_z_display_limits(ax)

        if cax is not None:
            cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
        else:
            cb = fig.colorbar(mesh, ax=ax)
        cb.set_label("rays")

        if axtop is not None:
            if top_stat.lower() == "fwhm":
                arr = np.asarray(caustic.get("fwhm", {}).get(axis_key, None), dtype=float)
                label = "FWHM [µm]"; scale = 1e6
            elif top_stat.lower() == "std":
                arr = np.asarray(caustic.get("moments", {}).get(axis_key, {}).get("std", None), dtype=float)
                label = "sigma [µm]"; scale = 1e6
            else:
                arr = None
            if arr is not None and arr.size == z.size:
                axtop.plot(z, arr * scale, lw=1.5)
                axtop.set_ylabel(label)
                axtop.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
                axtop.tick_params(direction="in", top=True, right=True)
                for spine in ("top", "right", "bottom", "left"):
                    axtop.spines[spine].set_visible(True)
                    axtop.spines[spine].set_color("black")
                axtop.set_xticklabels([])
                _apply_z_display_limits(axtop)

        out_path = path
        if path and which == "both":
            stem, ext = (path.rsplit(".", 1) + ["png"])[:2]
            suffix = "_x_vs_z" if axis_key == "x" else "_y_vs_z"
            out_path = f"{stem}{suffix}.{ext}"
        if out_path:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        return fig, ax

    results = []
    if which in ("x", "both"):
        results.append(_plot_one("x", np.asarray(Xmat, dtype=float)))
    if which in ("y", "both"):
        results.append(_plot_one("y", np.asarray(Ymat, dtype=float)))
    if plot:
        plt.show()
    return results if which == "both" else results[0]

def plot_energy(
    df: pd.DataFrame,
    *,
    bins: Optional[Union[int, Tuple[int, int]]] = None,   # int → auto for X; tuple ignored (kept for symmetry)
    bin_width: Optional[Number] = None,
    bin_method: int = 0,
    dpi: int = 100,
    path: Optional[str] = None,
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """energy distribution N vs E (eV), 1D histogram in counts."""
    fig_siz = 4.8

    if apply_style:
        start_plotting(k)

    df2 = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df
    e = pd.to_numeric(df2["energy"], errors="coerce").to_numpy(dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        fig, ax = plt.subplots(figsize=(fig_siz*6.4/4.8, fig_siz), dpi=dpi)
        ax.set_xlabel("energy [eV]")
        ax.set_ylabel("[rays]")
        ax.text(0.5, 0.5, "no finite energies", ha="center", va="center", transform=ax.transAxes)
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if plot:
            plt.show()
        return fig, ax

    nbx, _ = _auto_bins(e, e, bins, bin_width, bin_method)
    xr = _resolve_range(e, None)

    counts, edges = np.histogram(e, bins=nbx, range=xr)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(fig_siz*6.4/4.8, fig_siz), dpi=dpi)

    ax.fill_between(centers, 0, counts, step="mid", color="steelblue", alpha=0.5)

    ax.step(edges[:-1], counts, where="post", color="steelblue", linewidth=1.0)

    ax.set_xlim(xr)
    ax.set_ylim(0, 1.05 * max(1, counts.max()))
    ax.grid(which="major", linestyle="--", linewidth=0.3, color="dimgrey")
    ax.grid(which="minor", linestyle="--", linewidth=0.3, color="lightgrey")
    ax.set_xlabel("energy [eV]")
    ax.set_ylabel("[rays]")
    ax.locator_params(nbins=5)

    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if plot:
        plt.show()
    return fig, ax

def plot_energy_vs_intensity(
    df: pd.DataFrame,
    *,
    mode: str = "scatter",
    aspect_ratio: bool = False,
    color: Optional[int] = 3,
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    bins: BinsT = None,
    bin_width: Optional[Number] = None,
    bin_method: int = 0,
    dpi: int = 100,
    path: Optional[str] = None,
    showXhist: bool = True,
    showYhist: bool = True,
    envelope: bool = False,
    envelope_method: str = "edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]]:
    """2D plot with X=energy [eV], Y=Intensity [arb], with optional marginals/envelopes; never silently shows."""
    if apply_style:
        start_plotting(k)

    df2 = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df
    x = pd.to_numeric(df2["energy"], errors="coerce").to_numpy(dtype=float)         # eV
    y = pd.to_numeric(df2["intensity"], errors="coerce").to_numpy(dtype=float)      # [0,1]

    xl = r"energy [eV]"
    yl = r"$I$ [arb]"
    print(_resolve_mode(mode))
    fig, axes = _common_xy_plot(
        x, y, xl, yl,
        mode=_resolve_mode(mode),
        aspect_ratio=aspect_ratio,
        color=color,
        x_range=x_range,
        y_range=y_range,
        bins=bins,
        bin_width=bin_width,
        bin_method=bin_method,
        dpi=dpi,
        path=path,
        showXhist=showXhist,
        showYhist=showYhist,
        envelope=False,
        envelope_method=envelope_method,
    )
    if plot:
        plt.show()
    return fig, axes


# ---------------------------------------------------------------------------
# style settings
# ---------------------------------------------------------------------------

def start_plotting(k: float = 1.0) -> None:
    """
    Set global Matplotlib plot parameters scaled by factor k.

    Parameters
    ----------
    k : float, optional
        Scaling factor for font sizes (1.0 = baseline).
    """

    plt.rcParams.update(rcParamsDefault)

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "font.serif": ["Times New Roman"],
        "axes.grid": False,
        "savefig.bbox": "tight",
        "axes.spines.right": True,
        "axes.spines.top":   True,
    })

    plt.rc("axes",   titlesize=15. * k, labelsize=14 * k)
    plt.rc("xtick",  labelsize=13. * k)
    plt.rc("ytick",  labelsize=13. * k)
    plt.rc("legend", fontsize=13.* k)

    
@contextmanager
def plotting_style(k: float = 1.0):
    """
    Temporary plotting style (restores previous rcParams on exit).

    Examples
    --------
    >>> with plotting_style(1.2):
    ...     plot_beam(df)
    """
    old = plt.rcParams.copy()
    try:
        start_plotting(k)
        yield
    finally:
        plt.rcParams.update(old)

# ---------------------------------------------------------------------------
# private engine
# ---------------------------------------------------------------------------

def _common_xy_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    mode: ModeT,
    aspect_ratio: bool,
    color: Optional[int],
    x_range: RangeT,
    y_range: RangeT,
    bins: BinsT,
    bin_width: Optional[Number],
    bin_method: int,
    dpi: int,
    path: Optional[str],
    showXhist: bool = True,
    showYhist: bool = True,
    envelope: bool = True,
    envelope_method: Literal["edgeworth", "pearson", "maxent"] = "edgeworth",
) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]]:
    """Build core XY figure with central scatter/hist2d and optional 1D marginals/envelopes."""

    x_range = _resolve_range(x, x_range)
    y_range = _resolve_range(y, y_range)

    # if aspect_ratio is True:
    #     x_range = (np.min([x_range[0], y_range[0]]), np.max([x_range[1], y_range[1]]))
    #     y_range = x_range

    nb_of_bins = _auto_bins(x, y, bins, bin_width, bin_method)

    fig_siz = 6.4

    if aspect_ratio:
        fig_w, fig_h = fig_siz, fig_siz
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]
    else:
        fig_w, fig_h = fig_siz*6.4/4.8, fig_siz
        dx = fig_w
        dy = fig_h

    left, bottom, spacing = 0.20, 0.20, 0.02
    spacing_x, spacing_y = spacing, spacing
    kx = ky = k = 0.25

    if dx >= dy:
        width = 0.50
        height = width * dy / dx
        spacing_y = spacing * dy / dx
        ky = k * dy / dx
    else:
        height = 0.50
        width = height * dx / dy
        spacing_x = spacing * dx / dy
        kx = k * dx / dy

    rect_image = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing_x + 0.02, width, kx*.95]
    rect_histy = [left + width + spacing_x + 0.02, bottom, kx*.95, height]

    fig = plt.figure(figsize=(float(fig_w), float(fig_h)), dpi=int(dpi))
    ax_image = fig.add_axes(rect_image)
    ax_image.tick_params(top=False, right=False)
    ax_image.set_xlabel(x_label)
    ax_image.set_ylabel(y_label)

    ax_histx = ax_histy = None
    if showXhist:
        ax_histx = fig.add_axes(rect_histx, sharex=ax_image)
        ax_histx.tick_params(direction='in', which='both', labelbottom=False, top=True, right=True, colors='black')
        for sp in ('bottom', 'top', 'right', 'left'):
            ax_histx.spines[sp].set_color('black')
        ax_histx.hist(x, bins=nb_of_bins[0], range=x_range,
                    color='steelblue', linewidth=1, edgecolor='steelblue',
                    histtype='step', alpha=1)
        ax_histx.set_xlim(x_range)

        hx, _ = np.histogram(x, nb_of_bins[0], range=x_range)
        ax_histx.set_ylim(-0.05 * hx.max(), 1.05 * max(1, hx.max()))
        ax_histx.locator_params(tight=True, nbins=3)
        ax_histx.grid(which='major', linestyle='--', linewidth=0.3, color='dimgrey')
        ax_histx.grid(which='minor', linestyle='--', linewidth=0.3, color='lightgrey')
        ax_histx.set_ylabel('[rays]', fontsize='medium')
        if envelope:
            _overlay_envelope_on_hist(ax_histx, x, x_range, nb_of_bins[0],
                                    horizontal=False, method=envelope_method)

    if showYhist:
        ax_histy = fig.add_axes(rect_histy, sharey=ax_image)
        ax_histy.tick_params(direction='in', which='both', labelleft=False, top=True, right=True, colors='black')
        for sp in ('bottom', 'top', 'right', 'left'):
            ax_histy.spines[sp].set_color('black')
        ax_histy.hist(y, bins=nb_of_bins[1], range=y_range,
                    orientation='horizontal', color='steelblue',
                    linewidth=1, edgecolor='steelblue', histtype='step', alpha=1)
        ax_histy.set_ylim(y_range)
        hy, _ = np.histogram(y, nb_of_bins[1], range=y_range)
        ax_histy.set_xlim(-0.05 * hy.max(), 1.05 * max(1, hy.max()))
        ax_histy.locator_params(tight=True, nbins=3)
        ax_histy.grid(which='major', linestyle='--', linewidth=0.3, color='dimgrey')
        ax_histy.grid(which='minor', linestyle='--', linewidth=0.3, color='lightgrey')
        ax_histy.set_xlabel('[rays]', fontsize='medium')
        if envelope:
            _overlay_envelope_on_hist(ax_histy, y, y_range, nb_of_bins[1],
                                    horizontal=True, method=envelope_method)

    if mode == 'scatter':
        s, edgecolors, marker, linewidths = 2.5, 'face', '.', 1
        if color is None or color == 0:
            im = ax_image.scatter(x, y, color=_color_palette(0), alpha=1,
                                  edgecolors=edgecolors, s=s, marker=marker, linewidths=linewidths)
        else:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            z = z / z.max()
            cmap = _color_palette(color)
            clr = cmap(z)
            im = ax_image.scatter(x, y, color=clr, alpha=1, edgecolors=edgecolors,
                                  s=s, marker=marker, linewidths=linewidths)
        ax_image.grid(linestyle='--', linewidth=0.3, color='dimgrey')

    elif mode == 'hist2d':
        nbx, nby = nb_of_bins if isinstance(nb_of_bins, (tuple, list)) else (nb_of_bins, nb_of_bins)
        im = ax_image.hist2d(x, y, bins=[nbx, nby], cmap=_color_palette(color or 2))
    else:
        raise ValueError("mode must be 'scatter' or 'hist2d'.")

    ax_image.set_xlim(x_range)
    ax_image.set_ylim(y_range)

    ax_image.locator_params(tight=True, nbins=4)
    # ax_image.set_aspect('auto')
    ax_image.set_aspect('equal' if aspect_ratio else 'auto')

    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

    return fig, (ax_image, ax_histx, ax_histy)

def _prep_beam_xy(
    df: pd.DataFrame,
    *,
    kind: str,       
    direction: Optional[str] = None,
    z_offset: float = 0.0
):
    """Return (x, y, x_label, y_label) arrays scaled to µm or µrad, filtering alive rays."""


    if "lost_ray_flag" in df.columns:
        df = df.loc[df["lost_ray_flag"] == 0]

    if kind == "div":
        x = df["dX"].to_numpy(dtype=float) * 1e6
        y = df["dY"].to_numpy(dtype=float) * 1e6
        return x, y, r"$x'$ [$\mu$rad]", r"$y'$ [$\mu$rad]"

    X0 = df["X"].to_numpy(dtype=float)
    Y0 = df["Y"].to_numpy(dtype=float)
    dX = df["dX"].to_numpy(dtype=float)
    dY = df["dY"].to_numpy(dtype=float)

    if kind == "size":
        if z_offset != 0.0:
            Xp, Yp = _propagate_xy(X0, Y0, dX, dY, float(z_offset))
        else:
            Xp, Yp = X0, Y0
        return Xp * 1e6, Yp * 1e6, r"$x$ [$\mu$m]", r"$y$ [$\mu$m]"

    if kind == "ps":
        if direction not in {"x", "y"}:
            raise ValueError("direction must be 'x' or 'y' for phase space.")
        if direction == "x":
            Xp, _ = _propagate_xy(X0, Y0, dX, dY, float(z_offset)) if z_offset != 0.0 else (X0, Y0)
            pos = Xp * 1e6
            ang = dX * 1e6
        else:
            _, Yp = _propagate_xy(X0, Y0, dX, dY, float(z_offset)) if z_offset != 0.0 else (X0, Y0)
            pos = Yp * 1e6
            ang = dY * 1e6
        return pos, ang, (rf"${direction}$ [$\mu$m]"), (rf"${direction}'$ [$\mu$rad]")

    raise ValueError("kind must be one of {'size','div','ps'}.")

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _resolve_mode(mode: ModeT) -> Literal["scatter", "hist2d"]:
    """Normalize plotting mode/aliases and fallback to 'hist2d' with a warning."""
    if not isinstance(mode, str):
        warnings.warn(f"Plot mode {mode!r} not recognized (not a string). Falling back to 'hist2d'.")
        return "hist2d"
    m = mode.strip().lower()
    if m == "scatter" or m.startswith("s"):
        return "scatter"
    if m in {"histo", "hist2d", "histogram"} or m.startswith("h"):
        return "hist2d"
    warnings.warn(f"Plot mode {mode!r} not recognized. Falling back to 'hist2d'.")
    return "hist2d"

def _resolve_range(arr: np.ndarray, xr: RangeT) -> Tuple[float, float]:
    """Resolve (min,max) range with finite-data auto and 2% padding (safe for constant/empty arrays)."""
    if xr is not None and xr[0] is not None and xr[1] is not None:
        return (float(xr[0]), float(xr[1]))
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo == hi:
        pad = max(1e-12, abs(hi) * 0.02) or 1.0
        return (lo - pad, hi + pad)
    span = hi - lo
    pad = 0.02 * span
    return (lo - pad, hi + pad)

def _auto_bins(
    arrx: np.ndarray,
    arry: np.ndarray,
    bins: BinsT,
    bin_width: Optional[Number],
    bin_method: int,
) -> Tuple[int, int]:
    """Choose (nx, ny) bins via user value, width, or rule (sqrt/Sturges/Rice/Doane"""

    if bins is not None:
        return [bins, bins]

    bins = []

    for histos in [arrx, arry]:
        data = histos[np.isfinite(histos)]
        n = data.size
        if bin_width is not None:
            bins.append(int((np.amax(data)-np.amin(data))/bin_width))
        elif bin_method == -1:  # equidistribution
            nsgima = 6
            bwdth = np.std(data)/nsgima
            bins.append(int((np.amax(data)-np.amin(data))/bwdth))
        elif bin_method == 0:  # sqrt
            bins.append(int(np.sqrt(n)))
        elif bin_method == 1:  # Sturge
            bins.append(int(np.log2(n))+1)
        elif bin_method == 2:  # Rice
            bins.append(int(2*n**(1/3)))
        elif bin_method == 3:  # Doane's
            sigma_g1 = np.sqrt(6*(n-2)/((n+1)*(n+3)))
            bins.append(int(1+np.log2(n)*(1+moment(histos, order=3)/sigma_g1)))

    return bins

def _color_palette(color: Optional[int]) -> Union[Tuple[float, float, float], Colormap]:
    """Return a single RGB for monochrome scatter or a Matplotlib colormap for density/2D hist."""

    if color in (None, 0):
        return (0.0, 0.0, 0.0)
    if color == 1: return cm.viridis
    if color == 2: return cm.plasma
    if color == 3: return cm.turbo
    if color == 4: return cm.magma
    if color == 5: return cm.terrain
    # unknown: default to viridis as a safe colormap
    return cm.viridis

def _overlay_envelope_on_hist(ax, data, rng, nbins, *, horizontal=False,
                              method="edgeworth", color="darkred"):
    """Overlay a moment-matched PDF envelope onto a 1D histogram drawn in counts.

    We compute moments from samples, build a PDF on a fine axis, then scale by N*bin_width
    so the curve sits in 'counts' space.
    """
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return

    # moments
    mu, sig, skew, kurt = stats.calc_moments_from_particle_distribution(d)  # (mu,sigma,gamma1,gamma2_excess)
    if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0):
        return

    # axis to evaluate the envelope
    xmin, xmax = rng
    # be generous: mu+-6sigma but clipped to plotting range, and dense for a smooth curve
    lo = max(xmin, mu - 6*sig)
    hi = min(xmax, mu + 6*sig)
    axis = np.linspace(lo, hi, 1024)

    # envelope (PDF) on that axis
    env = stats.calc_envelope_from_moments(
        mean=mu, std=sig, skewness=skew, kurtosis_excess=kurt,
        axis=axis, method=method, clip_negative=True
    )["envelope"]

    # scale to histogram counts: counts \approx N * PDF * bin_width
    N = d.size
    bin_width = (xmax - xmin) / max(2, int(nbins))
    counts_curve = N * env * bin_width

    # plot
    if horizontal:
        ax.plot(counts_curve, axis, color=color, linewidth=0.5, alpha=1)
    else:
        ax.plot(axis, counts_curve, color=color, linewidth=0.5, alpha=1)

def _make_plane_centered_edges(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return np.array([0.0, 1.0])
    if z.size == 1:
        dz = 1e-3 if np.isfinite(z[0]) else 1.0
        return np.array([z[0] - 0.5*dz, z[0] + 0.5*dz])
    dz = np.diff(z)
    mids = z[:-1] + 0.5*dz
    first = z[0] - 0.5*dz[0]
    last  = z[-1] + 0.5*dz[-1]
    return np.concatenate([[first], mids, [last]])

def _edges_from_span(lo: float, hi: float, *, bin_width: Optional[Number], bins: Optional[int]) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if bin_width is not None:
        bw = float(bin_width)
        if bw <= 0:
            raise ValueError("bin_width must be positive")
        n = int(np.ceil((hi - lo) / bw))
        return np.linspace(lo, lo + n*bw, n+1)
    if bins is None:
        bins = 200
    return np.linspace(lo, hi, int(bins)+1)