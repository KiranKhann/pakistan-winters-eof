"""Figures: the paper's Figs. 1-3, plus the application and reconstruction figures.

Colour roles (one palette everywhere):
  series 1 / 2      blue #2a78d6, orange #eb6834      (PC1, PC2; identity)
  diverging         blue <-> grey <-> red              (signed fields: SLP, temperature, r)
  diverging, rain   brown <-> grey <-> teal            (precipitation: dry <-> wet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

from .eof import EOFResult

# --------------------------------------------------------------------------- #
# Palette and style
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#d03b3b"

DIV_COLORS = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec",
              "#f8d3cf", "#f09b93", "#e34948", "#b8302f", "#7a1a1a"]
RAIN_COLORS = ["#5e3c0f", "#8f5e1c", "#c48a3a", "#e3c08a", "#f3e3c8", "#f0efec",
               "#c9ebe1", "#84d1bd", "#3fa98f", "#1a7a66", "#0d4d42"]
CMAP_DIV = LinearSegmentedColormap.from_list("blue_grey_red", DIV_COLORS)
CMAP_RAIN = LinearSegmentedColormap.from_list("brown_grey_teal", RAIN_COLORS)

PC_COLORS = [BLUE, ORANGE]


def set_style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
    })


set_style()


def _sym_levels(field: np.ndarray, n: int = 10, q: float = 99.0, step: float | None = None) -> np.ndarray:
    """Contour levels symmetric about zero, spanning the q-th percentile of |field|.
    Zero itself is left out, so the central bin (-step, +step) straddles zero and
    takes the neutral grey midpoint of the diverging colour map."""
    vmax = float(np.nanpercentile(np.abs(field), q)) if np.isfinite(field).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    if step is None:
        raw = 2 * vmax / n
        mag = 10 ** np.floor(np.log10(raw))
        step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    top = step * np.ceil(vmax / step)
    levels = np.arange(-top, top + step / 2, step)
    return levels[np.abs(levels) > step * 1e-6]


CORR_LEVELS = np.array([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def _circle_boundary(ax) -> None:
    theta = np.linspace(0, 2 * np.pi, 120)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)


def polar_map(ax, field, lat, lon, levels, cmap=CMAP_DIV, lat_min: float = 20.0, lines: bool = True):
    """Filled contours on a north-polar stereographic axis (like the paper's Fig. 2)."""
    f_c, lon_c = add_cyclic_point(field, coord=lon)
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    cf = ax.contourf(lon_c, lat, f_c, levels=levels, cmap=cmap, norm=norm,
                     extend="both", transform=ccrs.PlateCarree())
    if lines:
        ax.contour(lon_c, lat, f_c, levels=[lv for lv in levels if abs(lv) > 1e-12],
                   colors=INK, linewidths=0.35, alpha=0.45, transform=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", color=INK_2, linewidth=0.5)
    ax.set_extent([-180, 180, lat_min, 90], crs=ccrs.PlateCarree())
    gl = ax.gridlines(color=GRID, linewidth=0.5, xlocs=range(-180, 181, 60), ylocs=[30, 60])
    gl.xlines = True
    _circle_boundary(ax)
    return cf


def region_map(ax, field, lat, lon, levels, extent, cmap=CMAP_DIV, pvals=None, alpha=0.05,
               outline=None):
    """Filled contours on a lat-lon map with optional stippling where p < alpha
    and an optional shapely geometry outlined (e.g. Pakistan)."""
    lon180 = ((lon + 180) % 360) - 180
    order = np.argsort(lon180)
    lon_s, f_s = lon180[order], field[:, order]
    norm = BoundaryNorm(levels, cmap.N, extend="both")
    cf = ax.contourf(lon_s, lat, f_s, levels=levels, cmap=cmap, norm=norm, extend="both",
                     transform=ccrs.PlateCarree())
    if pvals is not None:
        p_s = pvals[:, order]
        LO, LA = np.meshgrid(lon_s, lat)
        sig = np.isfinite(p_s) & (p_s < alpha)
        ax.scatter(LO[sig], LA[sig], s=3.5, c=INK, alpha=0.75, linewidths=0, transform=ccrs.PlateCarree(),
                   label=f"p < {alpha:g}")
    ax.coastlines(resolution="50m", color=INK_2, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=MUTED, linewidth=0.35)
    if outline is not None:
        ax.add_geometries([outline], crs=ccrs.PlateCarree(), facecolor="none", edgecolor=INK, linewidth=1.3)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, color=GRID, linewidth=0.5, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = dict(size=8, color=INK_2)
    return cf


# --------------------------------------------------------------------------- #
# Paper Figure 1: eigenvalue spectrum with North et al. error bars
# --------------------------------------------------------------------------- #
def fig_spectrum(res: EOFResult, kmax: int = 50, title: str | None = None):
    k = min(kmax, res.all_var_pct.size)
    rank = np.arange(1, k + 1)
    lam, err = res.all_var_pct[:k], res.all_north_err_pct[:k]
    sep = np.ones(res.all_var_pct.size, dtype=bool)
    lo, hi = res.all_var_pct - res.all_north_err_pct, res.all_var_pct + res.all_north_err_pct
    sep[1:] &= hi[1:] < lo[:-1]
    sep[:-1] &= lo[:-1] > hi[1:]
    sep = sep[:k]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(rank, lam, yerr=err, fmt="none", ecolor=BLUE, elinewidth=1.0, capsize=2.5, capthick=1.0)
    ax.scatter(rank[sep], lam[sep], s=30, color=BLUE, zorder=3, label="separated from neighbours")
    ax.scatter(rank[~sep], lam[~sep], s=30, facecolor=SURFACE, edgecolor=BLUE, linewidth=1.2, zorder=3,
               label="overlaps a neighbour (degenerate)")
    for i in range(min(2, k)):
        ax.annotate(f"EOF{i + 1}: {lam[i]:.1f}%", (rank[i], lam[i]), xytext=(10, 2),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlim(0, k + 1)
    ax.set_ylim(0, max(lam[0] + err[0], 1) * 1.12)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Eigenvalue (% of total variance)")
    ax.set_title(title or "Eigenvalue spectrum")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.text(1.0, -0.2, f"Error bars: North et al. (1982), eq. (12), n = {res.n_eff:.0f}",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Paper Figure 2: leading EOF patterns
# --------------------------------------------------------------------------- #
def fig_eof_maps(res: EOFResult, ks=(0, 1), kind: str = "regression", season_label: str | None = None,
                 var_label: str = "SLP"):
    """kind="regression": anomaly regressed on standardised PC (physical units per s.d.)
    kind="loading":    the unit-norm EOF vector itself (dimensionless)."""
    maps = [res.regression_map(k) if kind == "regression" else res.loading_map(k) for k in ks]
    levels = _sym_levels(np.stack(maps), n=12)
    units = res.meta.get("units", "") if kind == "regression" else "dimensionless"
    season = season_label or res.meta.get("season", "")

    fig = plt.figure(figsize=(4.2 * len(ks), 4.9))
    axes = [fig.add_subplot(1, len(ks), i + 1, projection=ccrs.NorthPolarStereo()) for i in range(len(ks))]
    cf = None
    for ax, k, m in zip(axes, ks, maps):
        cf = polar_map(ax, m, res.lat, res.lon, levels, lat_min=float(max(res.lat.min(), 0)))
        ax.set_title(f"{season} {var_label} EOF{k + 1} ({res.var_pct[k]:.0f}%)", loc="center")
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06, shrink=0.75)
    cb.set_label(f"{units} per standard deviation of the PC" if kind == "regression" else "EOF loading")
    cb.outline.set_visible(False)
    return fig


def fig_eof_maps_region(res: EOFResult, ks=(0, 1), extent=None, outline=None, title_prefix: str = "",
                        var_label: str = "SLP"):
    """Regression maps of regional EOFs on a lat-lon projection."""
    maps = [res.regression_map(k) for k in ks]
    levels = _sym_levels(np.stack(maps), n=10)
    if extent is None:
        lon180 = ((res.lon + 180) % 360) - 180
        extent = [float(lon180.min()), float(lon180.max()), float(res.lat.min()), float(res.lat.max())]
    fig = plt.figure(figsize=(5.2 * len(ks), 4.7))
    axes = [fig.add_subplot(1, len(ks), i + 1, projection=ccrs.PlateCarree()) for i in range(len(ks))]
    cf = None
    for ax, k, m in zip(axes, ks, maps):
        cf = region_map(ax, m, res.lat, res.lon, levels, extent, outline=outline)
        ax.set_title(f"{title_prefix}{var_label} EOF{k + 1} ({res.var_pct[k]:.0f}%)")
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, shrink=0.6)
    cb.set_label(f"{res.meta.get('units', '')} per standard deviation of the PC")
    cb.outline.set_visible(False)
    return fig


# --------------------------------------------------------------------------- #
# Paper Figure 3: standardised PCs with +/-1 extreme-event thresholds
# --------------------------------------------------------------------------- #
def _season_segments(time: pd.DatetimeIndex, season_year, values):
    """Insert NaN between seasons so a line joins months of the same winter only."""
    t, v = [], []
    prev = None
    for ti, sy, vi in zip(time, season_year, values):
        if prev is not None and sy != prev:
            t.append(ti - pd.Timedelta(days=1))
            v.append(np.nan)
        t.append(ti)
        v.append(vi)
        prev = sy
    return pd.DatetimeIndex(t), np.asarray(v)


def fig_pcs(res: EOFResult, ks=(0, 1), season_label: str | None = None, var_label: str = "SLP",
            threshold: float = 1.0, names=None):
    season = season_label or res.meta.get("season", "")
    sy = res.season_year if res.season_year is not None else res.time.year.values
    fig, axes = plt.subplots(len(ks), 1, figsize=(8.0, 2.6 * len(ks)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, k) in enumerate(zip(axes, ks)):
        color = PC_COLORS[i % len(PC_COLORS)]
        pc = res.pcs_std[:, k]
        tt, vv = _season_segments(res.time, sy, pc)
        ax.plot(tt, vv, color=color, linewidth=1.1)
        ext = np.abs(pc) > threshold
        ax.scatter(res.time[~ext], pc[~ext], s=9, color=color, zorder=3, linewidths=0)
        ax.scatter(res.time[ext], pc[ext], s=26, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=4,
                   label=f"|PC| > {threshold:g}  ({ext.sum()} months)")
        for y in (threshold, -threshold):
            ax.axhline(y, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_ylim(-3.6, 3.6)
        ax.set_ylabel(names[i] if names else f"PC{k + 1}")
        ax.set_title(f"{season} {var_label} PC{k + 1}")
        ax.legend(loc="upper left", fontsize=8.5)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Application figures
# --------------------------------------------------------------------------- #
def fig_correlation_maps(panels, lat, lon, extent, outline=None, suptitle=None):
    """panels: list of (title, r, p). Correlation maps with stippling where p < 0.05."""
    levels = CORR_LEVELS
    fig = plt.figure(figsize=(5.6 * len(panels), 4.6))
    axes = [fig.add_subplot(1, len(panels), i + 1, projection=ccrs.PlateCarree()) for i in range(len(panels))]
    cf = None
    for ax, (title, r, p) in zip(axes, panels):
        cf = region_map(ax, r, lat, lon, levels, extent, pvals=p, outline=outline)
        ax.set_title(title)
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, shrink=0.6)
    cb.set_label("Correlation with the AO index (dots: p < 0.05)")
    cb.outline.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, x=0.02, ha="left", fontsize=12, fontweight="bold")
    return fig


def fig_composite_maps(panels, lat, lon, extent, outline=None, suptitle=None):
    """panels: list of (title, diff, p, units, cmap)."""
    fig = plt.figure(figsize=(5.6 * len(panels), 4.9))
    for i, (title, diff, p, units, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection=ccrs.PlateCarree())
        lon180 = ((lon + 180) % 360) - 180
        e0, e1, e2, e3 = extent
        box = (lat >= e2) & (lat <= e3)
        boxl = (lon180 >= e0) & (lon180 <= e1)
        levels = _sym_levels(diff[np.ix_(box, boxl)], n=10)
        cf = region_map(ax, diff, lat, lon, levels, extent, cmap=cmap, pvals=p, outline=outline)
        ax.set_title(title)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.05, pad=0.1, shrink=0.85)
        cb.set_label(units)
        cb.outline.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, x=0.02, ha="left", fontsize=12, fontweight="bold")
    return fig


def fig_scatter(panels):
    """panels: list of (x, y, stats_dict, xlabel, ylabel, title)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (x, y, st, xl, yl, title) in zip(axes, panels):
        ax.scatter(x, y, s=22, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, st["intercept"] + st["slope"] * xs, color=INK, linewidth=1.4)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ptxt = "p < 0.001" if st["p"] < 0.001 else f"p = {st['p']:.3f}"
        ax.text(0.02, 0.97, f"r = {st['r']:+.2f}   {ptxt}\nslope = {st['slope']:+.2f} per s.d.   n = {st['n']}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK, zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9, pad=2.5))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_winter_bars(panels, title=None):
    """Stacked bar panels sharing the winter axis.
    panels: list of (series indexed by winter year, ylabel, (neg_color, pos_color))."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.0, 1.9 * len(panels) + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (s, yl, (cneg, cpos)) in zip(axes, panels):
        colors = np.where(s.values >= 0, cpos, cneg)
        ax.bar(s.index, s.values, width=0.72, color=colors)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("Winter (year of January)")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Reconstruction / compression (paper eq. 11)
# --------------------------------------------------------------------------- #
def fig_reconstruction(res: EOFResult, t: int, ks=(1, 2, 5, 10), var_label: str = "SLP"):
    orig = res.anom[t]
    recs = [res.reconstruct(k, t) for k in ks]
    levels = _sym_levels(orig, n=12)
    n = 1 + len(ks)
    fig = plt.figure(figsize=(3.3 * n, 4.2))
    axes = [fig.add_subplot(1, n, i + 1, projection=ccrs.NorthPolarStereo()) for i in range(n)]
    lat_min = float(max(res.lat.min(), 0))
    cf = polar_map(axes[0], orig, res.lat, res.lon, levels, lat_min=lat_min, lines=False)
    axes[0].set_title(f"Observed\n{res.time[t]:%b %Y}", loc="center")
    cos = np.cos(np.deg2rad(res.lat))[:, None] * np.ones((1, res.lon.size))
    v = res.valid
    for ax, k, rec in zip(axes[1:], ks, recs):
        polar_map(ax, rec, res.lat, res.lon, levels, lat_min=lat_min, lines=False)
        rmse = np.sqrt(np.nansum(((orig - rec) ** 2 * cos)[v]) / cos[v].sum())
        ax.set_title(f"{k} EOF{'s' if k > 1 else ''}\nRMSE {rmse:.2f} {res.meta.get('units', '')}", loc="center")
    cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.05, pad=0.05, shrink=0.6)
    cb.set_label(f"{var_label} anomaly ({res.meta.get('units', '')})")
    cb.outline.set_visible(False)
    return fig


def fig_rmse_curve(res: EOFResult, kmax: int = 40, marks=(1, 2, 5, 10, 20)):
    ks = np.arange(1, min(kmax, res.all_var_pct.size) + 1)
    rmse = res.rmse_curve(ks)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ks, rmse, color=BLUE)
    for m in marks:
        if m <= ks[-1]:
            ax.scatter([m], [rmse[m - 1]], s=26, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.annotate(f"{m}: {np.cumsum(res.all_var_pct)[m - 1]:.0f}% var, {res.compression_ratio(m):.0f}x smaller",
                        (m, rmse[m - 1]), xytext=(8, 6), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xlabel("Number of EOFs kept (k)")
    ax.set_ylabel(f"RMSE ({res.meta.get('units', '')})")
    ax.set_ylim(0, None)
    ax.set_title("Reconstruction error vs number of EOFs (eq. 11)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Plain-language map for the public page
# --------------------------------------------------------------------------- #
def fig_public_map(field, lat, lon, extent, title: str, low_label: str, high_label: str,
                   point=None, point_label: str | None = None, outline=None, vmax: float | None = None,
                   cmap=CMAP_DIV, width: float = 6.4):
    """A map with a colour bar that reads in words ("lower pressure" ... "higher pressure")
    instead of numbers alone, and an optional marker for the chosen place."""
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(field), 98)) or 1.0
    step = vmax / 4
    mag = 10 ** np.floor(np.log10(step))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= step), default=step)
    levels = _sym_levels(field, step=step)
    fig = plt.figure(figsize=(width, width * 0.78))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    cf = region_map(ax, field, lat, lon, levels, extent, cmap=cmap, outline=outline)
    if point is not None:
        ax.scatter([point[1]], [point[0]], s=70, marker="o", facecolor=INK, edgecolor=SURFACE, linewidth=1.8,
                   zorder=6, transform=ccrs.PlateCarree())
        if point_label:
            ax.annotate(point_label, (point[1], point[0]), xytext=(7, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=INK, zorder=7,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1.5),
                        transform=ccrs.PlateCarree(), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))
    ax.set_title(title)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, shrink=0.9)
    cb.outline.set_visible(False)
    cb.set_label("hPa", fontsize=8, color=MUTED, labelpad=1)
    cb.ax.text(0.0, -2.6, f"← {low_label}", transform=cb.ax.transAxes, ha="left", va="top", fontsize=9, color=INK_2)
    cb.ax.text(1.0, -2.6, f"{high_label} →", transform=cb.ax.transAxes, ha="right", va="top", fontsize=9, color=INK_2)
    return fig
