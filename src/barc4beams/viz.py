# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2025 Synchrotron SOLEIL

"""
viz.py - plotting routines for beams and beamline layouts.
"""

from __future__ import annotations

from contextlib import contextmanager
from itertools import cycle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParamsDefault
from matplotlib.gridspec import GridSpec

Number = Union[int, float]
BinsT = Union[int, Tuple[int, int], None]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_beam(
    df: pd.DataFrame,
    *,
    mode: str = "scatter",
    aspect_ratio: bool = True,
    color_scheme: Optional[Union[int, str]] = None,
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    bins: BinsT = None,
    show_marginals: bool = True,
    dpi: int = 300,
    save: Optional[str] = None,
    apply_style: bool = True,
    k: float = 1.0,
    **kwargs,
):
    """
    Plot the spatial footprint of a standardized beam (X vs Y).

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized beam (from ``to_standard_beam``). Rays with
        ``lost_ray_flag==1`` are filtered out.
    mode : {'scatter', 'hist2d'}, optional
        Plot style: scatter plot of points, or 2D histogram. Default 'scatter'.
    aspect_ratio : bool, optional
        If True (default), enforce equal aspect ratio.
    color_scheme : int or str, optional
        Colormap to use. Integer values are mapped to default palettes,
        or pass a valid Matplotlib colormap name.
    x_range, y_range : tuple(float,float), optional
        Axis limits. If None, autoscale with padding.
    bins : int or (int,int), optional
        Number of bins for hist2d. If None, auto-chosen.
    show_marginals : bool, optional
        If True (default), show 1D histograms on top and right.
    dpi : int, optional
        Figure DPI (export resolution). Default 300.
    save : str, optional
        If provided, save the figure to this path.
    apply_style : bool, optional
        If True (default), apply :func:`start_plotting(k)` to set fonts/sizes.
    k : float, optional
        Font scaling factor for :func:`start_plotting`. Default 1.0.
    **kwargs
        Passed to the scatter routine (e.g. `s`, `alpha`).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of Axes
    """
    if apply_style:
        start_plotting(k)

    x, y, xl, yl = _prep_beam_xy(df, kind="size")
    return _common_xy_plot(
        x, y, xl, yl, mode=mode, aspect_ratio=aspect_ratio,
        color_scheme=color_scheme, x_range=x_range, y_range=y_range,
        bins=bins, show_marginals=show_marginals, dpi=dpi, save=save, **kwargs
    )


def plot_divergence(
    df: pd.DataFrame,
    *,
    mode: str = "scatter",
    aspect_ratio: bool = True,
    color_scheme: Optional[Union[int, str]] = None,
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    bins: BinsT = None,
    show_marginals: bool = True,
    dpi: int = 300,
    save: Optional[str] = None,
    apply_style: bool = True,
    k: float = 1.0,
    **kwargs,
):
    """
    Plot the angular footprint of a standardized beam (dX vs dY).

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized beam. Filters ``lost_ray_flag==1``.
    mode : {'scatter', 'hist2d'}, optional
        Plot style. Default 'scatter'.
    aspect_ratio : bool, optional
        If True (default), enforce equal aspect ratio.
    color_scheme : int or str, optional
        Colormap to use.
    x_range, y_range : tuple(float,float), optional
        Axis limits. If None, autoscale with padding.
    bins : int or (int,int), optional
        Number of bins for hist2d. Auto if None.
    show_marginals : bool, optional
        Show 1D histograms (default True).
    dpi : int, optional
        Figure DPI. Default 300.
    save : str, optional
        Save figure to this path.
    apply_style : bool, optional
        Apply :func:`start_plotting(k)` if True. Default True.
    k : float, optional
        Font scaling factor for :func:`start_plotting`. Default 1.0.
    **kwargs
        Extra scatter options.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of Axes

    Notes
    -----
    If ``direction='both'``, two panels (X–dX and Y–dY) are drawn side-by-side.
    In this case, ``show_marginals`` is ignored.
    """
    if apply_style:
        start_plotting(k)

    x, y, xl, yl = _prep_beam_xy(df, kind="div")
    return _common_xy_plot(
        x, y, xl, yl, mode=mode, aspect_ratio=aspect_ratio,
        color_scheme=color_scheme, x_range=x_range, y_range=y_range,
        bins=bins, show_marginals=show_marginals, dpi=dpi, save=save, **kwargs
    )


def plot_phase_space(
    df: pd.DataFrame,
    *,
    direction: str = "both",
    mode: str = "scatter",
    aspect_ratio: bool = True,
    color_scheme: Optional[Union[int, str]] = None,
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    bins: BinsT = None,
    show_marginals: bool = True,
    dpi: int = 300,
    save: Optional[str] = None,
    apply_style: bool = True,
    k: float = 1.0,
    **kwargs,
):
    """
    Plot the phase space of a standardized beam (X vs dX or Y vs dY).

    Parameters
    ----------
    df : pandas.DataFrame
        Standardized beam. Filters ``lost_ray_flag==1``.
    direction : {'x','y','both'}, optional
        Axis to plot. 'both' plots X–dX and Y–dY side-by-side. Default 'both'.
    mode : {'scatter', 'hist2d'}, optional
        Plot style. Default 'scatter'.
    aspect_ratio : bool, optional
        Equal aspect ratio if True (default).
    color_scheme : int or str, optional
        Colormap.
    x_range, y_range : tuple(float,float), optional
        Axis limits. If None, autoscale.
    bins : int or (int,int), optional
        Bins for hist2d. Auto if None.
    show_marginals : bool, optional
        Show 1D histograms (default True).
    dpi : int, optional
        Figure DPI. Default 300.
    save : str, optional
        Save figure to this path.
    apply_style : bool, optional
        Apply :func:`start_plotting(k)` if True. Default True.
    k : float, optional
        Font scaling factor for :func:`start_plotting`. Default 1.0.
    **kwargs
        Extra scatter args.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of Axes
    """
    if apply_style:
        start_plotting(k)

    direction = (direction or "both").lower()
    if direction not in {"x", "y", "both"}:
        raise ValueError("direction must be 'x', 'y', or 'both'.")

    if direction in {"x", "y"}:
        x, y, xl, yl = _prep_beam_xy(df, kind="ps", direction=direction)
        return _common_xy_plot(
            x, y, xl, yl, mode=mode, aspect_ratio=aspect_ratio,
            color_scheme=color_scheme, x_range=x_range, y_range=y_range,
            bins=bins, show_marginals=show_marginals, dpi=dpi, save=save, **kwargs
        )

    # two-panel layout (X–dX, Y–dY)
    x1, y1, xl1, yl1 = _prep_beam_xy(df, kind="ps", direction="x")
    x2, y2, xl2, yl2 = _prep_beam_xy(df, kind="ps", direction="y")

    fig = plt.figure(figsize=(10, 4), dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25, figure=fig)

    ax_main_left = fig.add_subplot(gs[0, 0])
    _draw_xy(
        ax_main_left, x1, y1, xl1, yl1, mode=mode, color_scheme=color_scheme,
        x_range=x_range, y_range=y_range, bins=bins, show_marginals=show_marginals,
        aspect_ratio=aspect_ratio, **kwargs
    )

    ax_main_right = fig.add_subplot(gs[0, 1])
    _draw_xy(
        ax_main_right, x2, y2, xl2, yl2, mode=mode, color_scheme=color_scheme,
        x_range=x_range, y_range=y_range, bins=bins, show_marginals=show_marginals,
        aspect_ratio=aspect_ratio, **kwargs
    )

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, (ax_main_left, ax_main_right)

def plot_beamline(shadow_info: Dict,
                  show_source: bool = True,
                  draw_to_scale: bool = False,
                  k: float = 1.0) -> None:
    """
    Plot beamline layout from SHADOW coordinates.

    Parameters
    ----------
    shadow_info : dict
        Output from `parse_shadow_info`:
          - "x", "y", "z" : np.ndarray [m]
          - "oe": {"labels": list[str], "type": list[str]} with {'S','E','M','G','O'}
    show_source : bool, optional
        If False, omit the source (first element). Default True.
    draw_to_scale : bool, optional
        If True, draw separate top/side figures (10x10) with equal metric scale AND
        vertical extent set to match the horizontal y-extent. Default False.
    k : float, optional
        Font scaling factor (passed to your styling if available). Default 1.0.
    """
    try:
        start_plotting(k)
    except NameError:
        pass

    x = np.asarray(shadow_info["x"], dtype=float)
    y = np.asarray(shadow_info["y"], dtype=float)
    z = np.asarray(shadow_info["z"], dtype=float)
    labels: List[str] = shadow_info.get("oe", {}).get("labels", [""] * len(x))
    types:  List[str] = shadow_info.get("oe", {}).get("type",   ["O"] * len(x))

    idxs = [i for i, t in enumerate(types) if t != "E" and (show_source or not (i == 0 and t == "S"))]
    if not idxs:
        raise ValueError("Nothing to plot after filtering (no elements).")

    color_map  = {"S": "darkred", "M": "olive", "G": "steelblue", "O": "teal"}
    marker_map = {"S": "*",       "M": "s",     "G": "D",         "O": "o"}
    legend_map = {"S": "source",  "M": "mirror","G": "grating",   "O": "screen"}

    def style(ax, ylabel):
        ax.set_facecolor("white")
        ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_ylabel(ylabel)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_color("black")

    def plot_points(ax, which: str, add_legend=True):
        """Plot markers for selected elements on ax versus y. which='top' (x) or 'side' (z)."""
        V = x if which == "top" else z
        ax.plot(y[idxs], V[idxs], color="0.6", lw=0.8, zorder=1)

        seen = set()
        for i in idxs:
            t = types[i]
            c = color_map.get(t, "black")
            m = marker_map.get(t, "o")
            lbl = legend_map[t] if (add_legend and t not in seen) else None
            seen.add(t)
            if t == "O":
                ax.plot(y[i], V[i], linestyle="none", marker=m,
                        markerfacecolor=c, markeredgecolor=c,
                        fillstyle="left", markersize=8, zorder=3, label=lbl)
            else:
                ax.plot(y[i], V[i], linestyle="none", marker=m,
                        markerfacecolor=("white" if t in ("M", "G") else c),
                        markeredgecolor=c, markeredgewidth=1.2,
                        markersize=8, zorder=3, label=lbl)

        if add_legend:
            h, _ = ax.get_legend_handles_labels()
            if h:
                ax.legend(loc="best", frameon=True)

    def square_limits_match_x(ax, Y: np.ndarray, V: np.ndarray):
        """
        Make the numeric span square with the *horizontal* span as master:
        vertical span (V) == horizontal y-span, centered on data midpoints.
        """
        ymin, ymax = np.nanmin(Y), np.nanmax(Y)
        yspan = max(ymax - ymin, 1e-9)
        yctr  = 0.5 * (ymax + ymin)

        vmin, vmax = np.nanmin(V), np.nanmax(V)
        vctr = 0.5 * (vmax + vmin)

        pad = 0.02 * yspan
        ax.set_xlim(yctr - 0.5*yspan - pad, yctr + 0.5*yspan + pad)
        ax.set_ylim(vctr - 0.5*yspan - pad, vctr + 0.5*yspan + pad)
        ax.set_aspect("equal", adjustable="box")

    if draw_to_scale:
        fig_top, ax_top = plt.subplots(figsize=(12, 12))
        fig_top.suptitle("Beamline layout — Top view (to scale)", fontsize=16 * k)
        style(ax_top, "top view [m]")
        plot_points(ax_top, which="top", add_legend=True)
        ax_top.set_xlabel("[m]")
        square_limits_match_x(ax_top, y[idxs], x[idxs])

        fig_side, ax_side = plt.subplots(figsize=(12, 12))
        fig_side.suptitle("Beamline layout — Side view (to scale)", fontsize=16 * k)
        style(ax_side, "side view [m]")
        plot_points(ax_side, which="side", add_legend=False)
        ax_side.set_xlabel("[m]")
        square_limits_match_x(ax_side, y[idxs], z[idxs])

        plt.show()

    else:
        fig, (ax_top, ax_side) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]}
        )
        fig.suptitle("Beamline layout", fontsize=16 * k, x=0.5)

        style(ax_top, "top view [m]")
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")

        plot_points(ax_top, which="top", add_legend=True)
        plot_points(ax_side, which="side", add_legend=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def plot_beamline_configs(configs: Sequence[Dict],
                          config_labels: Sequence[str],
                          show_source: bool = True,
                          draw_to_scale: bool = False,
                          k: float = 1.0) -> None:
    """
    Overlay multiple beamline configurations in the same plot_beamline-style graph.

    Parameters
    ----------
    configs : sequence of dict
        Each is the output of `parse_shadow_info`, with:
          - "x","y","z": np.ndarray
          - "oe": {"labels": list[str], "type": list[str]} where type in {'S','E','M','G','O'}
    config_labels : sequence of str
        REQUIRED. One label per configuration (used in the legend).
        Must be the same length as `configs`.
    show_source : bool, optional
        If False, omit 'S' at index 0 from every configuration. Default True.
    draw_to_scale : bool, optional
        If True, makes two separate square figures (10x10) for top/side views and
        forces the vertical span to match the horizontal y-span (combined across configs).
        If False, a single 12x6 figure with two stacked subplots sharing x (y). Default False.
    k : float, optional
        Font scaling hook for your styling. Default 1.0.
    """

    try:
        start_plotting(k)
    except NameError:
        pass

    if not configs:
        raise ValueError("`configs` is empty.")
    if len(config_labels) != len(configs):
        raise ValueError(f"`config_labels` must have length {len(configs)}, got {len(config_labels)}.")

    n_cfg = len(configs)
    cfg_labels = list(map(str, config_labels))

    cfg_colors_base = ["darkred", "olive", "steelblue", "teal", "peru", "slategray"]
    cfg_colors_iter = cycle(cfg_colors_base)
    cfg_colors = [next(cfg_colors_iter) for _ in range(n_cfg)]

    marker_map = {"S": "*", "M": "s", "G": "D", "O": "o"}

    def style(ax, ylabel):
        ax.set_facecolor("white")
        ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_ylabel(ylabel)
        for spine in ("top", "right", "bottom", "left"):
            ax.spines[spine].set_color("black")

    def filter_indices(types: List[str]) -> List[int]:
        return [i for i, t in enumerate(types) if t != "E" and (show_source or not (i == 0 and t == "S"))]

    Y_all_top, Vx_all, Y_all_side, Vz_all = [], [], [], []
    per_cfg_data = []
    for cfg in configs:
        x = np.asarray(cfg["x"], dtype=float)
        y = np.asarray(cfg["y"], dtype=float)
        z = np.asarray(cfg["z"], dtype=float)
        types = cfg.get("oe", {}).get("type", ["O"] * len(x))

        idxs = filter_indices(types)
        if not idxs:
            per_cfg_data.append((None, None, None, None))
            continue

        y_i = y[idxs]; x_i = x[idxs]; z_i = z[idxs]
        t_i = [types[i] for i in idxs]

        per_cfg_data.append((y_i, x_i, z_i, t_i))

        Y_all_top.append(y_i);  Vx_all.append(x_i)
        Y_all_side.append(y_i); Vz_all.append(z_i)

    def combined_span(Ys, Vs):
        Ys = [a for a in Ys if a is not None and len(a)]
        Vs = [a for a in Vs if a is not None and len(a)]
        if not Ys:
            return (-1, 1), (-1, 1)
        Yc = np.concatenate(Ys); Vc = np.concatenate(Vs)
        ymin, ymax = float(np.nanmin(Yc)), float(np.nanmax(Yc))
        yspan = max(ymax - ymin, 1e-9)
        yctr  = 0.5 * (ymax + ymin)
        vmin, vmax = float(np.nanmin(Vc)), float(np.nanmax(Vc))
        vctr  = 0.5 * (vmax + vmin)
        pad = 0.02 * yspan
        xlim = (yctr - 0.5*yspan - pad, yctr + 0.5*yspan + pad)
        ylim = (vctr - 0.5*yspan - pad, vctr + 0.5*yspan + pad)
        return xlim, ylim

    def plot_one(ax, which: str, add_legend: bool):
        """Plot all configs on a single axes (which='top' uses x, 'side' uses z)."""
        for (data, color, label) in zip(per_cfg_data, cfg_colors, cfg_labels):
            y_i, x_i, z_i, t_i = data
            if y_i is None:
                continue
            V = x_i if which == "top" else z_i

            ax.plot(y_i, V, color=color, lw=1.0, alpha=0.75, zorder=1,
                    label=(label if add_legend else None))

            for yy, vv, tt in zip(y_i, V, t_i):
                m = marker_map.get(tt, "o")
                if tt == "O":
                    ax.plot(yy, vv, linestyle="none", marker=m,
                            markerfacecolor=color, markeredgecolor=color,
                            fillstyle="left", markersize=8, zorder=3)
                elif tt in ("M", "G"):
                    ax.plot(yy, vv, linestyle="none", marker=m,
                            markerfacecolor="white", markeredgecolor=color,
                            markeredgewidth=1.2, markersize=8, zorder=3)
                else:  # 'S' or fallback
                    ax.plot(yy, vv, linestyle="none", marker=m,
                            markerfacecolor=color, markeredgecolor=color,
                            markersize=9 if tt == "S" else 8, zorder=3)

        if add_legend:
            h, _ = ax.get_legend_handles_labels()
            if h:
                ax.legend(loc="best", frameon=True)

    if draw_to_scale:
        xlim_top, ylim_top = combined_span(Y_all_top, Vx_all)
        xlim_side, ylim_side = combined_span(Y_all_side, Vz_all)

        fig_top, ax_top = plt.subplots(figsize=(10, 10))
        fig_top.suptitle("Beamline configs — Top view (to scale)", fontsize=16 * k)
        style(ax_top, "top view [m]")
        ax_top.set_xlabel("[m]")
        plot_one(ax_top, which="top", add_legend=True)
        ax_top.set_xlim(*xlim_top); ax_top.set_ylim(*ylim_top)
        ax_top.set_aspect("equal", adjustable="box")

        fig_side, ax_side = plt.subplots(figsize=(10, 10))
        fig_side.suptitle("Beamline configs — Side view (to scale)", fontsize=16 * k)
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")
        plot_one(ax_side, which="side", add_legend=False)
        ax_side.set_xlim(*xlim_side); ax_side.set_ylim(*ylim_side)
        ax_side.set_aspect("equal", adjustable="box")

        plt.show()

    else:
        fig, (ax_top, ax_side) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]}
        )
        fig.suptitle("Beamline configs", fontsize=16 * k, x=0.5)

        style(ax_top, "top view [m]")
        style(ax_side, "side view [m]")
        ax_side.set_xlabel("[m]")

        plot_one(ax_top, which="top", add_legend=True)
        plot_one(ax_side, which="side", add_legend=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

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

    plt.rc("axes",   titlesize=15 * k, labelsize=14 * k)
    plt.rc("xtick",  labelsize=13 * k)
    plt.rc("ytick",  labelsize=13 * k)
    plt.rc("legend", fontsize=12 * k)

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

def _prep_beam_xy(
    df: pd.DataFrame,
    *,
    kind: str,                   # 'size' | 'div' | 'ps'
    direction: Optional[str] = None,  # for kind='ps'
):
    """Return (x, y, x_label, y_label) arrays scaled to µm or µrad, filtering alive rays."""
    # Filter alive rays (0 = alive, 1 = lost)
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


def _common_xy_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    *,
    mode: str,
    aspect_ratio: bool,
    color_scheme: Optional[Union[int, str]],
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]],
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]],
    bins: BinsT,
    show_marginals: bool,
    dpi: int,
    save: Optional[str],
    **kwargs,
):
    """Common 2D plot builder with optional marginals."""
    fig = plt.figure(figsize=(5.5, 5.0), dpi=dpi)

    if show_marginals:
        # Layout: top marginal (short), right marginal (narrow)
        gs = GridSpec(2, 2, width_ratios=[4, 1.1], height_ratios=[1.1, 4],
                      hspace=0.05, wspace=0.05, figure=fig)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        # Hide tick labels on the shared axes
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
    else:
        ax_main = fig.add_subplot(111)
        ax_top = ax_right = None

    _draw_xy(
        ax_main, x, y, x_label, y_label, mode=mode, color_scheme=color_scheme,
        x_range=x_range, y_range=y_range, bins=bins,
        show_marginals=show_marginals, aspect_ratio=aspect_ratio, **kwargs
    )

    if show_marginals:
        _draw_marginals(ax_top, ax_right, x, y, x_label, y_label, x_range, y_range)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, (ax_main, ax_top, ax_right) if show_marginals else (ax_main,)


def _draw_xy(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    *,
    mode: str,
    color_scheme: Optional[Union[int, str]],
    x_range: Optional[Tuple[Optional[Number], Optional[Number]]],
    y_range: Optional[Tuple[Optional[Number], Optional[Number]]],
    bins: BinsT,
    show_marginals: bool,
    aspect_ratio: bool,
    **kwargs,
):
    """Render main 2D content (scatter or hist2d)."""
    # Axis cosmetics (simple, readable)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

    # Limits (auto with light padding if unspecified)
    xr = _resolve_range(x, x_range)
    yr = _resolve_range(y, y_range)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)

    # Aspect
    if aspect_ratio:
        ax.set_aspect("equal", adjustable="box")

    # Labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Colors
    cmap = _map_color_scheme(color_scheme)

    if mode == "hist2d":
        # Auto bins if needed
        if bins is None:
            bx = _auto_bins(x, xr)
            by = _auto_bins(y, yr)
            bins_ = (bx, by)
        elif isinstance(bins, int):
            bins_ = (bins, bins)
        else:
            bins_ = bins

        H, xedges, yedges = np.histogram2d(x[np.isfinite(x)], y[np.isfinite(y)], bins=bins_, range=[xr, yr])
        # Use pcolormesh for sharper edges
        X, Y = np.meshgrid(xedges, yedges, indexing="xy")
        im = ax.pcolormesh(X, Y, H.T, cmap=cmap, shading="auto")
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("counts")

    elif mode == "scatter":
        s = kwargs.get("s", 1.0)
        alpha = kwargs.get("alpha", 1.0)
        ax.scatter(x, y, s=s, alpha=alpha, marker=".", linewidths=0, edgecolors="none", c="k" if cmap is None else None, cmap=cmap)
    else:
        raise ValueError("mode must be 'scatter' or 'hist2d'.")


def _draw_marginals(
    ax_top,
    ax_right,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    x_range: Tuple[Optional[Number], Optional[Number]],
    y_range: Tuple[Optional[Number], Optional[Number]],
):
    """Top and right 1D histograms (densities)."""
    # Top (X)
    ax_top.tick_params(direction="in", labelbottom=False)
    ax_top.hist(x[np.isfinite(x)], bins=_auto_bins(x, x_range), range=x_range, color="0.2")
    ax_top.set_ylabel("counts")

    # Right (Y)
    ax_right.tick_params(direction="in", labelleft=False)
    ax_right.hist(y[np.isfinite(y)], bins=_auto_bins(y, y_range), range=y_range, orientation="horizontal", color="0.2")
    ax_right.set_xlabel("counts")

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _resolve_range(arr: np.ndarray, xr: Optional[Tuple[Optional[Number], Optional[Number]]]):
    """Resolve (xmin, xmax): if None values, auto from finite data with 2% padding."""
    if xr is not None and xr[0] is not None and xr[1] is not None:
        return xr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = np.min(finite), np.max(finite)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        pad = 1.0 if not np.isfinite(lo) or not np.isfinite(hi) else max(1e-12, abs(hi) * 0.02)
        return (float(lo - pad), float(hi + pad))
    span = hi - lo
    pad = 0.02 * span
    return (float(lo - pad), float(hi + pad))


def _auto_bins(a: np.ndarray, xr: Tuple[Optional[Number], Optional[Number]]):
    """Freedman–Diaconis like bin heuristic, clamped to [50, 300]."""
    data = a[np.isfinite(a)]
    n = data.size
    if n < 2:
        return 50
    q75, q25 = np.percentile(data, [75, 25])
    iqr = max(1e-12, q75 - q25)
    span = (xr[1] - xr[0]) if xr and xr[0] is not None and xr[1] is not None else (data.max() - data.min())
    bw = 2.0 * iqr * n ** (-1.0 / 3.0)
    bins = int(max(10, min(300, round(span / max(bw, 1e-12)))))
    return bins


def _map_color_scheme(color_scheme: Optional[Union[int, str]]):
    """Map legacy integer schemes to mpl colormaps. None → default cycle / black scatter."""
    if color_scheme is None:
        return None
    if isinstance(color_scheme, str):
        return plt.get_cmap(color_scheme)
    # legacy integer mapping (tweak as you wish)
    mapping = {
        1: "Greys",
        2: "viridis",
        3: "plasma",
        4: "magma",
        5: "cividis",
    }
    return plt.get_cmap(mapping.get(int(color_scheme), "viridis"))