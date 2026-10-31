"""Numerical checks that the implementation satisfies the properties derived in
Hannachi (2004), Section 3. Each check prints PASS or FAIL.

    python scripts/verify_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eofkit import run_eof_analysis  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str) -> None:
    global failures
    failures += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    res = run_eof_analysis("slp", n_years=30)
    U, s, Vt = res._svd
    vmask = res.valid.reshape(-1)
    w = res.weights.reshape(-1)[vmask]
    Xw = res.anom.reshape(res.n, -1)[:, vmask] * w
    print(f"Project window: n = {res.n} months, p = {res.p_used} grid points used\n")

    # Sec. 2: anomalies have zero time mean at every grid point (eq. 4)
    m = np.abs(res.anom.reshape(res.n, -1)[:, vmask].mean(axis=0)).max()
    check("anomalies are centred (eq. 4)", m < 1e-8, f"max |column mean| = {m:.1e} hPa")

    # Sec. 3.2.2: EOFs are orthonormal
    G = Vt[:20] @ Vt[:20].T
    e = np.abs(G - np.eye(20)).max()
    check("EOFs orthonormal (a_i . a_j = delta_ij)", e < 1e-10, f"max error = {e:.1e}")

    # Sec. 3.2.2: PCs are uncorrelated
    C = np.corrcoef(res.pcs.T)
    e = np.abs(C - np.eye(C.shape[0])).max()
    check("PCs uncorrelated", e < 1e-8, f"max |off-diagonal corr| = {e:.1e}")

    # eq. (9) vs eq. (13): SVD gives the eigenvalues of the covariance matrix
    lam_eig = np.sort(np.linalg.eigvalsh(Xw @ Xw.T / (res.n - 1)))[::-1][:20]
    e = np.abs(lam_eig - res.all_var_pct[:20] * res.total_var / 100).max() / lam_eig[0]
    check("SVD eigenvalues = eigenvalues of Sigma (eq. 9, 13)", e < 1e-10, f"max relative error = {e:.1e}")

    # eq. (8)-(9): Sigma a_k = lambda_k a_k, checked without forming the p x p matrix
    a1 = Vt[0]
    Sa = Xw.T @ (Xw @ a1) / (res.n - 1)
    e = np.linalg.norm(Sa - res.eigvals[0] * a1) / res.eigvals[0]
    check("Sigma a_1 = lambda_1 a_1 (eq. 9)", e < 1e-10, f"relative residual = {e:.1e}")

    # eq. (10): PC = projection of the weighted anomalies on the EOF
    e = np.abs(Xw @ Vt[0] - res.pcs[:, 0]).max() / np.abs(res.pcs[:, 0]).max()
    check("PC_1 = X'_w a_1 (eq. 10)", e < 1e-10, f"max relative error = {e:.1e}")

    # eq. (11): using every EOF reconstructs the data exactly
    full = res.reconstruct(res.all_var_pct.size)
    e = np.nanmax(np.abs(full - res.anom)[:, res.valid])
    check("full expansion reconstructs X' exactly (eq. 11)", e < 1e-8, f"max error = {e:.1e} hPa")

    # Explained variances sum to 100%
    check("explained variances sum to 100%", abs(res.all_var_pct.sum() - 100) < 1e-9,
          f"sum = {res.all_var_pct.sum():.12f}")

    # Eckart-Young: rank-k error equals the discarded eigenvalues
    k = 5
    Xk = (U[:, :k] * s[:k]) @ Vt[:k]
    err = np.sum((Xw - Xk) ** 2)
    tail = np.sum(s[k:] ** 2)
    check("rank-5 error = sum of discarded sigma^2 (Eckart-Young)", abs(err - tail) / tail < 1e-10,
          f"{err:.6e} vs {tail:.6e}")

    # Paper Sec. 3.5 numbers on the paper's own sample
    paper = run_eof_analysis("slp", time_range=("1948-01", "2000-12"))
    ok = abs(paper.var_pct[0] - 21) < 1 and abs(paper.var_pct[1] - 13) < 1
    check("paper's 21% / 13% reproduced (1948-2000)", ok,
          f"{paper.var_pct[0]:.1f}% / {paper.var_pct[1]:.1f}%")

    print(f"\n{'All checks passed.' if failures == 0 else f'{failures} check(s) failed.'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
