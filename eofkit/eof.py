"""Empirical Orthogonal Functions following Hannachi (2004), Sections 2 and 3.

Equation numbers in comments refer to the paper.

    (1)-(2)  F(t, lat, lon)  ->  data matrix X (n x p), p = p1*p2
    (3)-(4)  anomalies X' = X - climatology
    (5)-(6)  area weighting  X'_w = X' D,  D = diag(cos(lat))
    (7)      covariance  Sigma = X'_w^T X'_w / (n-1)
    (8)-(9)  EOFs = eigenvectors of Sigma, eigenvalues = explained variance
    (10)     principal components  c_k = X'_w a_k
    (11)     truncated reconstruction  X' ~ sum_k c_k a_k^T
    (12)     North et al. (1982) rule of thumb  d(lambda_k) ~ lambda_k sqrt(2/n)
    (13)     SVD  X'_w = U S V^T  (we never form Sigma explicitly)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr


# --------------------------------------------------------------------------- #
# Section 2: structure of climate data
# --------------------------------------------------------------------------- #
def anomalies(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Eq. (3)-(4): remove the seasonal cycle, computed as the mean of each
    calendar month over the analysis window ("monthly averaged values then
    removed", paper Sec. 3.5). Returns (anomaly, climatology)."""
    clim = da.groupby("time.month").mean("time")
    anom = (da.groupby("time.month") - clim).drop_vars("month", errors="ignore")
    anom.attrs.update(da.attrs)
    return anom, clim


def area_weights(da: xr.DataArray, scheme: str = "cos") -> xr.DataArray:
    """Eq. (5): the diagonal weighting matrix D as a (lat, lon) field.

    scheme="cos"      D = diag(cos phi)            eq. (5) exactly as printed
    scheme="sqrtcos"  D = diag(sqrt(cos phi))      the covariance (7) is then weighted
                      by cos phi, i.e. by grid-cell area. This is the scheme that
                      reproduces the paper's own numbers (21% / 13% for 1948-2000),
                      so it is the project default.
    scheme="none"     no weighting (shows the polar artefact the paper warns about)
    """
    w_lat = np.cos(np.deg2rad(da.lat.values)).clip(min=0.0)
    if scheme == "sqrtcos":
        w_lat = np.sqrt(w_lat)
    elif scheme == "none":
        w_lat = np.ones_like(w_lat)
    elif scheme != "cos":
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    w2d = np.broadcast_to(w_lat[:, None], (da.lat.size, da.lon.size)).copy()
    return xr.DataArray(w2d, coords={"lat": da.lat, "lon": da.lon}, dims=("lat", "lon"))


def to_matrix(anom: xr.DataArray) -> tuple[np.ndarray, xr.DataArray]:
    """Eq. (1)-(2): concatenate the two spatial dimensions,
    F (n, p1, p2) -> X (n, p). Equivalent to the paper's MATLAB reshape."""
    stacked = anom.transpose("time", "lat", "lon").stack(space=("lat", "lon"))
    return np.asarray(stacked.values, dtype=float), stacked


def effective_sample_size(x: np.ndarray) -> float:
    """Heuristic effective sample size n (1 - r1) / (1 + r1) from the lag-1
    autocorrelation r1 (paper footnote 2: n should count independent samples)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    r1 = float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else 0.0
    r1 = min(max(r1, -0.99), 0.99)
    return float(x.size * (1.0 - r1) / (1.0 + r1))


# --------------------------------------------------------------------------- #
# Section 3: EOFs
# --------------------------------------------------------------------------- #
@dataclass
class EOFResult:
    """Everything produced by one EOF analysis.

    Spatial arrays are (k, p1, p2) or (p1, p2); time arrays are (n, k) or (n,).
    """
    eofs: np.ndarray            # (k, p1, p2) unit-norm loadings in weighted space (NaN = unused)
    pcs: np.ndarray             # (n, k) raw PCs  c_k = X'_w a_k               eq. (10)
    pcs_std: np.ndarray         # (n, k) standardised PCs (zero mean, unit variance)
    eigvals: np.ndarray         # (k,)  lambda_k
    var_pct: np.ndarray         # (k,)  100 lambda_k / sum lambda
    north_err: np.ndarray       # (k,)  d lambda_k = lambda_k sqrt(2 / n_eff)  eq. (12)
    north_err_pct: np.ndarray   # (k,)  same, in % of total variance
    all_var_pct: np.ndarray     # every non-zero eigenvalue in %, for the spectrum plot
    all_north_err_pct: np.ndarray
    total_var: float
    n: int
    p: int
    p_used: int
    n_eff: float
    time: pd.DatetimeIndex
    season_year: np.ndarray | None
    lat: np.ndarray
    lon: np.ndarray
    weights: np.ndarray         # (p1, p2) the diagonal of D
    valid: np.ndarray           # (p1, p2) bool, columns that entered the SVD
    anom: np.ndarray            # (n, p1, p2) unweighted anomalies (physical units)
    meta: dict = field(default_factory=dict)
    _svd: tuple = field(default=None, repr=False)   # (U, s, Vt) restricted to valid columns

    # ----- spatial patterns ------------------------------------------------------
    def loading_map(self, k: int, unweight: bool = False) -> np.ndarray:
        """The k-th EOF a_k as a map (unit vector in the weighted space).
        unweight=True divides by the weights to show the pattern in the
        original (unweighted) coordinates."""
        pat = self.eofs[k].copy()
        if unweight:
            with np.errstate(divide="ignore", invalid="ignore"):
                pat = np.where(self.weights > 1e-8, pat / self.weights, np.nan)
        return pat

    def regression_map(self, k: int) -> np.ndarray:
        """Anomaly field regressed on standardised PC k: the pattern in physical
        units per one standard deviation of the PC (hPa per s.d. for SLP).
        This is how EOF maps are usually drawn for interpretation."""
        X = self.anom.reshape(self.n, -1)
        c = self.pcs_std[:, k]
        return (X.T @ c / (self.n - 1)).reshape(self.lat.size, self.lon.size)

    # ----- eq. (11): truncated reconstruction --------------------------------------
    def reconstruct(self, k: int, t: int | None = None) -> np.ndarray:
        """Reconstruct the (unweighted) anomaly field from the leading k EOFs.
        Returns (n, p1, p2), or (p1, p2) when a time index t is given."""
        U, s, Vt = self._svd
        rows = slice(None) if t is None else slice(t, t + 1)
        Xw_hat = (U[rows, :k] * s[:k]) @ Vt[:k]                     # eq. (11), weighted space
        vmask = self.valid.reshape(-1)
        w = self.weights.reshape(-1)[vmask]
        out = np.full((Xw_hat.shape[0], self.lat.size * self.lon.size), np.nan)
        out[:, vmask] = Xw_hat / w                                   # undo D
        out = out.reshape(-1, self.lat.size, self.lon.size)
        return out[0] if t is not None else out

    def rmse_curve(self, ks) -> np.ndarray:
        """Area-weighted RMSE (physical units) of the rank-k reconstruction for
        each k in ks, computed over all n time steps."""
        vmask = self.valid.reshape(-1)
        cosw = (np.cos(np.deg2rad(self.lat))[:, None] * np.ones((1, self.lon.size))).reshape(-1)[vmask]
        X = self.anom.reshape(self.n, -1)[:, vmask]
        U, s, Vt = self._svd
        w = self.weights.reshape(-1)[vmask]
        out = []
        for k in ks:
            Xhat = ((U[:, :k] * s[:k]) @ Vt[:k]) / w
            err2 = ((X - Xhat) ** 2 * cosw).sum() / (cosw.sum() * self.n)
            out.append(float(np.sqrt(err2)))
        return np.asarray(out)

    def compression_ratio(self, k: int) -> float:
        """Storage of the full matrix (n p) over storage of the rank-k factors k (n + p)."""
        return (self.n * self.p_used) / (k * (self.n + self.p_used))

    # ----- eq. (12): is EOF k separable from its neighbours? -------------------------
    def separable(self) -> np.ndarray:
        """True when the North error bar of eigenvalue k does not overlap the bars
        of its neighbours k-1 and k+1 (otherwise the EOFs are degenerate / mixed)."""
        lam, err = self.all_var_pct, self.all_north_err_pct
        lo, hi = lam - err, lam + err
        ok = np.ones(lam.size, dtype=bool)
        ok[1:] &= hi[1:] < lo[:-1]
        ok[:-1] &= lo[:-1] > hi[1:]
        return ok[: self.eigvals.size]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "EOF": np.arange(1, self.eigvals.size + 1),
            "eigenvalue": self.eigvals,
            "variance_%": self.var_pct,
            "cumulative_%": np.cumsum(self.var_pct),
            "north_error_%": self.north_err_pct,
            "separable": self.separable(),
        })

    def flip_sign(self, k: int) -> None:
        """EOFs are defined only up to sign (paper Sec. 3.4). Flip EOF k and its PC."""
        self.eofs[k] *= -1
        self.pcs[:, k] *= -1
        self.pcs_std[:, k] *= -1
        U, s, Vt = self._svd
        U[:, k] *= -1
        Vt[k] *= -1


def compute_eofs(anom: xr.DataArray, weights: xr.DataArray, neofs: int = 20,
                 n_eff: float | None = None, meta: dict | None = None) -> EOFResult:
    """EOF analysis of an anomaly field by SVD (paper Sec. 3.2-3.3).

    anom     (time, lat, lon) anomalies, physical units
    weights  (lat, lon) diagonal of D from area_weights()
    neofs    number of EOFs/PCs to retain (all eigenvalues are kept for the spectrum)
    n_eff    effective sample size for eq. (12); defaults to n, as in the paper
    """
    X, _ = to_matrix(anom)                               # (n, p)        eq. (2)
    n, p = X.shape
    w = np.asarray(weights.transpose("lat", "lon").values, dtype=float).reshape(-1)

    # Columns that can enter the analysis: no missing values and non-zero weight.
    # The pole row has cos(90 deg) = 0 and would otherwise be a column of zeros.
    valid = np.isfinite(X).all(axis=0) & (w > 1e-8)
    Xw = X[:, valid] * w[valid]                          # X'_w = X' D   eq. (6)

    # eq. (13): thin SVD. Rows of Vt are the EOFs and U*s are the PCs. Because
    # p >> n this costs O(n^2 p) instead of the O(p^3) eigen-decomposition of Sigma.
    U, s, Vt = np.linalg.svd(Xw, full_matrices=False)

    lam = s ** 2 / (n - 1)                               # eigenvalues of Sigma  eq. (7)-(9)
    keep = lam > 1e-12 * lam[0]                          # drop the numerically-zero tail
    lam, U, s, Vt = lam[keep], U[:, keep], s[keep], Vt[keep]
    total = float(lam.sum())
    n_eff = float(n if n_eff is None else n_eff)
    err = lam * np.sqrt(2.0 / n_eff)                     # eq. (12)

    k = int(min(neofs, lam.size))
    pcs = U[:, :k] * s[:k]                               # c_k = X'_w a_k   eq. (10)
    pcs_std = pcs / pcs.std(axis=0, ddof=1)

    eofs = np.full((k, p), np.nan)
    eofs[:, valid] = Vt[:k]

    p1, p2 = anom.lat.size, anom.lon.size
    season_year = anom["season_year"].values if "season_year" in anom.coords else None

    res = EOFResult(
        eofs=eofs.reshape(k, p1, p2),
        pcs=pcs, pcs_std=pcs_std,
        eigvals=lam[:k], var_pct=100.0 * lam[:k] / total,
        north_err=err[:k], north_err_pct=100.0 * err[:k] / total,
        all_var_pct=100.0 * lam / total, all_north_err_pct=100.0 * err / total,
        total_var=total, n=n, p=p, p_used=int(valid.sum()), n_eff=n_eff,
        time=pd.DatetimeIndex(anom.time.values), season_year=season_year,
        lat=anom.lat.values.copy(), lon=anom.lon.values.copy(),
        weights=w.reshape(p1, p2), valid=valid.reshape(p1, p2),
        anom=np.asarray(anom.transpose("time", "lat", "lon").values, dtype=float),
        meta=dict(meta or {}),
        _svd=(U.copy(), s.copy(), Vt.copy()),
    )
    # Generic sign convention: the grid point with the largest |loading| is positive.
    # Application code may re-orient specific EOFs (e.g. the AO convention).
    for j in range(k):
        flat = res.eofs[j].reshape(-1)
        if flat[np.nanargmax(np.abs(flat))] < 0:
            res.flip_sign(j)
    return res
