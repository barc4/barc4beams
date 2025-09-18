# SPDX-License-Identifier: GPL-3.0-only
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
    dpi: int = 300,
    path: Optional[str] = None,
    showXhist=True,
    showYhist=True,
    envelope=True,
    envelope_method="edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True
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

    x, y, xl, yl = _prep_beam_xy(df, kind="size")
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
    dpi: int = 300,
    path: Optional[str] = None,
    showXhist=True,
    showYhist=True,
    envelope=True,
    envelope_method="edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True

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

    x, y, xl, yl = _prep_beam_xy(df, kind="div")
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
    dpi: int = 300,
    path: Optional[str] = None,
    showXhist=True,
    showYhist=True,
    envelope=True,
    envelope_method="edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True

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
        x, y, xl, yl = _prep_beam_xy(df, kind="ps", direction=d)
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
    mode: str = "hist2d",                 # 'scatter' | 'hist2d' (aliases accepted)
    aspect_ratio: bool = False,           # z vs X/Y typically very different scales
    color: Optional[int] = 2,
    z_range: RangeT = None,
    xy_range: RangeT = None,
    bins: BinsT = None,
    bin_width: Optional[Number] = None,
    bin_method: int = 0,
    dpi: int = 300,
    path: Optional[str] = None,           # if provided, auto-add _x/_y suffix when which='both'
    showXhist: bool = True,               # hist over z
    showYhist: bool = True,               # hist over X or Y
    envelope: bool = False,               # envelopes are less meaningful on z; default False
    envelope_method: str = "edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True
):
    """
    Plot caustics as 2D scatter/hist2d:
      - X vs z (µm vs m)
      - Y vs z (µm vs m)

    Parameters mirror the other viz helpers for consistency.
    """
    if apply_style:
        start_plotting(k)

    z = np.asarray(caustic["optical_axis"], dtype=float)
    cm = caustic.get("caustic", {})
    Xmat = cm.get("X", None)
    Ymat = cm.get("Y", None)

    def _stack(mat):
        """Return stacked (z_stacked, pos_um) for a matrix of shape (P, N)."""
        if mat is None:
            return None, None
        mat = np.asarray(mat, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != z.size:
            raise ValueError("caustic['caustic'][X/Y] must have shape (len(z), N).")
        P, N = mat.shape
        z_stacked = np.repeat(z, N)
        pos_um    = mat.reshape(P * N) * 1e6
        return z_stacked, pos_um

    def _suffix(base: Optional[str], suf: str) -> Optional[str]:
        if not base:
            return None
        stem, ext = (base.rsplit(".", 1) + ["png"])[:2]
        return f"{stem}{suf}.{ext}"

    out = []

    if which in ("x", "both"):
        zx, xu = _stack(Xmat)
        if zx is None:
            raise ValueError("X matrix not present in caustic['caustic']['X'].")
        fig_x, axes_x = _common_xy_plot(
            x=zx, y=xu,
            x_label=r"$z$ [m]", y_label=r"$x$ [$\mu$m]",
            mode=_resolve_mode(mode),
            aspect_ratio=aspect_ratio,
            color=color,
            x_range=z_range,
            y_range=xy_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=_suffix(path, "_x_vs_z") if (which == "both") else path,
            showXhist=showXhist,
            showYhist=showYhist,
            envelope=envelope,
            envelope_method=envelope_method,
        )
        out.append((fig_x, axes_x))

    if which in ("y", "both"):
        zy, yu = _stack(Ymat)
        if zy is None:
            raise ValueError("Y matrix not present in caustic['caustic']['Y'].")
        fig_y, axes_y = _common_xy_plot(
            x=zy, y=yu,
            x_label=r"$z$ [m]", y_label=r"$y$ [$\mu$m]",
            mode=_resolve_mode(mode),
            aspect_ratio=aspect_ratio,
            color=color,
            x_range=z_range,
            y_range=xy_range,
            bins=bins,
            bin_width=bin_width,
            bin_method=bin_method,
            dpi=dpi,
            path=_suffix(path, "_y_vs_z") if (which == "both") else path,
            showXhist=showXhist,
            showYhist=showYhist,
            envelope=envelope,
            envelope_method=envelope_method,
        )
        out.append((fig_y, axes_y))

    if plot:
        plt.show()

    if which == "both":
        return tuple(out)
    return out[0]

def plot_energy(
    df: pd.DataFrame,
    *,
    bins: Optional[Union[int, Tuple[int, int]]] = None,   # int → auto for X; tuple ignored (kept for symmetry)
    bin_width: Optional[Number] = None,
    bin_method: int = 0,
    dpi: int = 300,
    path: Optional[str] = None,
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Energy distribution N vs E (eV), 1D histogram in counts."""
    fig_siz = 3

    if apply_style:
        start_plotting(k)

    # filter alive rays
    df2 = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df
    e = pd.to_numeric(df2["energy"], errors="coerce").to_numpy(dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        fig, ax = plt.subplots(figsize=(fig_siz*6.4/4.8, fig_siz), dpi=dpi)
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("[counts]")
        ax.text(0.5, 0.5, "no finite energies", ha="center", va="center", transform=ax.transAxes)
        if path:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if plot:
            plt.show()
        return fig, ax

    # binning
    nbx, _ = _auto_bins(e, e, bins, bin_width, bin_method)
    xr = _resolve_range(e, None)

    # compute histogram
    counts, edges = np.histogram(e, bins=nbx, range=xr)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(fig_siz*6.4/4.8, fig_siz), dpi=dpi)

    # filled area
    ax.fill_between(centers, 0, counts, step="mid", color="steelblue", alpha=0.5)

    # crisp outline
    ax.step(edges[:-1], counts, where="post", color="steelblue", linewidth=1.0)

    ax.set_xlim(xr)
    ax.set_ylim(0, 1.05 * max(1, counts.max()))
    ax.grid(which="major", linestyle="--", linewidth=0.3, color="dimgrey")
    ax.grid(which="minor", linestyle="--", linewidth=0.3, color="lightgrey")
    ax.set_xlabel("Energy [eV]")
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
    mode: str = "scatter",              # 'scatter' | 'histo2d' (aliases accepted)
    aspect_ratio: bool = False,         # default False: ranges are typically very unequal
    color: Optional[int] = 3,
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    bins: BinsT = None,
    bin_width: Optional[Number] = None,
    bin_method: int = 0,
    dpi: int = 300,
    path: Optional[str] = None,
    showXhist: bool = True,
    showYhist: bool = True,
    envelope: bool = False,              # envelope is meaningful on Energy; on Intensity it's bounded in [0,1]
    envelope_method: str = "edgeworth",
    apply_style: bool = True,
    k: float = 1.0,
    plot: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes], Optional[plt.Axes]]]:
    """2D plot with X=Energy [eV], Y=Intensity [arb], with optional marginals/envelopes; never silently shows."""
    if apply_style:
        start_plotting(k)

    # alive rays only
    df2 = df.loc[df["lost_ray_flag"] == 0] if "lost_ray_flag" in df.columns else df
    x = pd.to_numeric(df2["energy"], errors="coerce").to_numpy(dtype=float)         # eV
    y = pd.to_numeric(df2["intensity"], errors="coerce").to_numpy(dtype=float)      # [0,1]
    # labels
    xl = r"$E$ [eV]"
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
    })

    plt.rc("axes",   titlesize=12. * k, labelsize=12. * k)
    plt.rc("xtick",  labelsize=11. * k)
    plt.rc("ytick",  labelsize=11. * k)
    plt.rc("legend", fontsize=11.* k)

    plt.rcParams.update({
        "axes.grid": False,
        "savefig.bbox": "tight",
        "axes.spines.right": True,
        "axes.spines.top":   True,
    })

    
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

    if aspect_ratio is True:
        x_range = (np.min([x_range[0], y_range[0]]), np.max([x_range[1], y_range[1]]))
        y_range = x_range

    nb_of_bins = _auto_bins(x, y, bins, bin_width, bin_method)

    fig_siz = 3
    # --- figure & rectangles (your math, lightly tidied) ---
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

    # --- histograms ---
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
        ax_histy.set_xlim(-0.05 * hx.max(), 1.05 * max(1, hy.max()))
        ax_histy.locator_params(tight=True, nbins=3)
        ax_histy.grid(which='major', linestyle='--', linewidth=0.3, color='dimgrey')
        ax_histy.grid(which='minor', linestyle='--', linewidth=0.3, color='lightgrey')
        ax_histy.set_xlabel('[rays]', fontsize='medium')
        if envelope:
            _overlay_envelope_on_hist(ax_histy, y, y_range, nb_of_bins[1],
                                    horizontal=True, method=envelope_method)
    # --- main scatter / hist2d ---
    ax_image.set_xlim(x_range)
    ax_image.set_ylim(y_range)

    if mode == 'scatter':
        s, edgecolors, marker, linewidths = 1, 'face', '.', 0.1
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
                                  s=s, marker=marker, linewidths=linewidths * 2)
        ax_image.grid(linestyle='--', linewidth=0.3, color='dimgrey')

    elif mode == 'hist2d':
        nbx, nby = nb_of_bins if isinstance(nb_of_bins, (tuple, list)) else (nb_of_bins, nb_of_bins)
        im = ax_image.hist2d(x, y, bins=[nbx, nby], cmap=_color_palette(color or 2))
    else:
        raise ValueError("mode must be 'scatter' or 'hist2d'.")

    # ticks/aspect
    ax_image.locator_params(tight=True, nbins=4)
    ax_image.set_aspect('auto')
    # ax_image.set_aspect('equal' if aspect_ratio else 'auto')

    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')

    return fig, (ax_image, ax_histx, ax_histy)

def _prep_beam_xy(
    df: pd.DataFrame,
    *,
    kind: str,       
    direction: Optional[str] = None,
):
    """Return (x, y, x_label, y_label) arrays scaled to µm or µrad, filtering alive rays."""

    if "lost_ray_flag" in df.columns:
        df = df.loc[df["lost_ray_flag"] == 0]

    if kind == "size":
        x = df["X"].to_numpy(dtype=float) * 1e6
        y = df["Y"].to_numpy(dtype=float) * 1e6
        return x, y, r"$x$ [$\mu$m]", r"$y$ [$\mu$m]"

    if kind == "div":
        x = df["dX"].to_numpy(dtype=float) * 1e6
        y = df["dY"].to_numpy(dtype=float) * 1e6
        return x, y, r"$x'$ [$\mu$rad]", r"$y'$ [$\mu$rad]"

    if kind == "ps":
        if direction not in {"x", "y"}:
            raise ValueError("direction must be 'x' or 'y' for phase space.")
        c = direction.upper()
        pos = df[c].to_numpy(dtype=float) * 1e6
        ang = df[f"d{c}"].to_numpy(dtype=float) * 1e6
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
    mu, sig, skew, kurt = stats.calc_moments_from_particle_distribution(d)  # (μ,σ,γ1,γ2_excess)
    if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0):
        return

    # axis to evaluate the envelope
    xmin, xmax = rng
    # be generous: μ±6σ but clipped to plotting range, and dense for a smooth curve
    lo = max(xmin, mu - 6*sig)
    hi = min(xmax, mu + 6*sig)
    axis = np.linspace(lo, hi, 1024)

    # envelope (PDF) on that axis
    env = stats.calc_envelope_from_moments(
        mean=mu, std=sig, skewness=skew, kurtosis_excess=kurt,
        axis=axis, method=method, clip_negative=True
    )["envelope"]

    # scale to histogram counts: counts ≈ N * PDF * bin_width
    N = d.size
    bin_width = (xmax - xmin) / max(2, int(nbins))
    counts_curve = N * env * bin_width

    # plot
    if horizontal:
        ax.plot(counts_curve, axis, color=color, linewidth=0.5, alpha=1)
    else:
        ax.plot(axis, counts_curve, color=color, linewidth=0.5, alpha=1)