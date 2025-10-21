
#!/usr/bin/env python3
import argparse
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ----------------- Config -----------------
@dataclass
class WLSConfig:
    el_min_deg: float = 12.0
    cn0_min_dbhz: float = 20.0
    max_iters: int = 6
    tol_dx_m: float = 1e-4
    robust: bool = True
    huber_delta_m: float = 2.0
    a_m: float = 0.5
    b_m: float = 4.0
    c_m: float = 100.0

# ----------------- Geodesy -----------------
_A = 6378137.0
_F = 1.0/298.257223563
_E2 = _F*(2 - _F)
_B = _A * (1 - _F)

def ecef_to_llh(xyz: np.ndarray) -> Tuple[float,float,float]:
    x, y, z = xyz
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    E2p = (_A**2 - _B**2) / _B**2
    t = math.atan2(z * _A, r * _B)
    st, ct = math.sin(t), math.cos(t)
    lat = math.atan2(z + E2p * _B * st**3, r - _E2 * _A * ct**3)
    s = math.sin(lat)
    N = _A / math.sqrt(1 - _E2 * s*s)
    h = r / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), h

def ecef_to_enu_cov(P_ecef: np.ndarray, llh: Tuple[float,float,float]) -> np.ndarray:
    lat, lon, _ = llh
    lat = math.radians(lat); lon = math.radians(lon)
    sl, cl = math.sin(lat), math.cos(lat)
    sln, cln = math.sin(lon), math.cos(lon)
    R = np.array([
        [-sln,            cln,           0.0],
        [-sl*cln, -sl*sln,       cl],
        [ cl*cln,  cl*sln,       sl]
    ], dtype=float)
    return R @ P_ecef @ R.T

# ----------------- Weights -----------------
def sigma2_model(cn0_dbhz: float, el_rad: float, a: float, b: float, c: float) -> float:
    s = max(math.sin(el_rad), 1e-3)
    return a*a + (b/s)**2 + c * 10**(-cn0_dbhz/10.0)

# ----------------- Core WLS -----------------
@dataclass
class SatResidual:
    sv: str
    el_deg: float
    az_deg: float
    cn0_dbhz: float
    weight: float
    residual_m: float

@dataclass
class CoarseWLSSolution:
    unix_ms: int
    ecef_xyz: np.ndarray
    llh: tuple
    clock_bias_m: float
    nsats_used: int
    pdop: float
    cov_ecef: np.ndarray
    cov_enu: np.ndarray
    sigmaE: float
    sigmaN: float
    sigmaU: float
    resid: List[SatResidual]
    valid: bool = True

def solve_epoch_wls_from_rows(rows: pd.DataFrame,
                              seed_xyz: Optional[np.ndarray],
                              seed_clk_m: float,
                              cfg: WLSConfig) -> CoarseWLSSolution:
    # Filter sats by elevation & CN0
    mask = (rows["SvElevationDegrees"] >= cfg.el_min_deg) & (rows["Cn0DbHz"] >= cfg.cn0_min_dbhz)
    sats_df = rows[mask].copy()
    unix_ms = int(sats_df["utcTimeMillis"].iloc[0]) if len(sats_df) else int(rows["utcTimeMillis"].iloc[0])
    if len(sats_df) < 5:
        return CoarseWLSSolution(
            unix_ms=unix_ms, ecef_xyz=np.full(3, np.nan), llh=(np.nan,np.nan,np.nan),
            clock_bias_m=np.nan, nsats_used=int(len(sats_df)), pdop=np.nan,
            cov_ecef=np.full((3,3), np.nan), cov_enu=np.full((3,3), np.nan),
            sigmaE=np.nan, sigmaN=np.nan, sigmaU=np.nan, resid=[], valid=False
        )

    # Prepare arrays
    sat_xyz = sats_df[["SvPositionXEcefMeters","SvPositionYEcefMeters","SvPositionZEcefMeters"]].to_numpy(float)
    pr = sats_df["RawPseudorangeMeters"].to_numpy(float)
    cn0 = sats_df["Cn0DbHz"].to_numpy(float)
    el_rad = np.radians(sats_df["SvElevationDegrees"].to_numpy(float))
    az_deg = sats_df["SvAzimuthDegrees"].to_numpy(float)
    sv_id = sats_df["Svid"].astype(str).tolist()

    trop = sats_df["TroposphericDelayMeters"].fillna(0.0).to_numpy(float) if "TroposphericDelayMeters" in sats_df else np.zeros(len(sats_df))
    iono = sats_df["IonosphericDelayMeters"].fillna(0.0).to_numpy(float) if "IonosphericDelayMeters" in sats_df else np.zeros(len(sats_df))
    sv_clk = sats_df["SvClockBiasMeters"].fillna(0.0).to_numpy(float) if "SvClockBiasMeters" in sats_df else np.zeros(len(sats_df))

    # Seed
    X = seed_xyz.copy() if seed_xyz is not None else sat_xyz.mean(axis=0) * 0.0
    Cb = float(seed_clk_m)

    # Iterative WLS
    w_list = None
    for it in range(cfg.max_iters):
        H = []; v = []; w = []
        for i in range(len(sats_df)):
            sat = sat_xyz[i]
            rho_vec = sat - X
            rho = float(np.linalg.norm(rho_vec))
            u = rho_vec / max(rho, 1e-6)
            pred = rho + Cb + trop[i] + iono[i] + sv_clk[i]
            v_i = pr[i] - pred
            H.append([-u[0], -u[1], -u[2], 1.0])
            v.append(v_i)
            w.append(1.0 / sigma2_model(cn0[i], el_rad[i], cfg.a_m, cfg.b_m, cfg.c_m))
        H = np.asarray(H, float)
        v = np.asarray(v, float)
        W = np.diag(w)
        A = H.T @ W @ H
        b = H.T @ W @ v
        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return CoarseWLSSolution(
                unix_ms=unix_ms, ecef_xyz=np.full(3, np.nan), llh=(np.nan,np.nan,np.nan),
                clock_bias_m=np.nan, nsats_used=int(len(sats_df)), pdop=np.nan,
                cov_ecef=np.full((3,3), np.nan), cov_enu=np.full((3,3), np.nan),
                sigmaE=np.nan, sigmaN=np.nan, sigmaU=np.nan, resid=[], valid=False
            )
        X += dx[:3]; Cb += dx[3]
        if np.linalg.norm(dx[:3]) < cfg.tol_dx_m:
            w_list = w
            break

        if cfg.robust:
            r = v - H @ dx
            med = np.median(r)
            s = 1.4826 * np.median(np.abs(r - med)) + 1e-6
            z = r / s
            delta = cfg.huber_delta_m
            mult = np.ones_like(z)
            mask = np.abs(z) > delta
            mult[mask] = (delta / np.abs(z[mask]))
            W = np.diag(w) @ np.diag(mult)
            A = H.T @ W @ H
            b = H.T @ W @ v
            try:
                dx2 = np.linalg.solve(A, b)
                X += dx2[:3]; Cb += dx2[3]
            except np.linalg.LinAlgError:
                pass
        w_list = w

    # Covariance & metrics
    r = v - H @ dx
    dof = max(len(sats_df)-4, 1)
    W = np.diag(w_list)
    sigma0_2 = float(r.T @ (W @ r) / dof)
    P4 = np.linalg.inv(H.T @ W @ H) * sigma0_2
    P_ecef = P4[:3,:3]
    llh = ecef_to_llh(X)
    P_enu = ecef_to_enu_cov(P_ecef, llh)
    sigE, sigN, sigU = np.sqrt(np.clip(np.diag(P_enu), 0.0, np.inf))
    Q = np.linalg.inv(H.T @ H)
    pdop = float(math.sqrt(np.trace(Q[:3,:3])))

    resid_list = []
    for i in range(len(sats_df)):
        resid_list.append(SatResidual(
            sv=str(sv_id[i]),
            el_deg=float(np.degrees(el_rad[i])),
            az_deg=float(az_deg[i]),
            cn0_dbhz=float(cn0[i]),
            weight=float(w_list[i]),
            residual_m=float(r[i])
        ))

    return CoarseWLSSolution(
        unix_ms=unix_ms, ecef_xyz=X, llh=llh, clock_bias_m=Cb,
        nsats_used=int(len(sats_df)), pdop=pdop,
        cov_ecef=P_ecef, cov_enu=P_enu,
        sigmaE=float(sigE), sigmaN=float(sigN), sigmaU=float(sigU),
        resid=resid_list, valid=True
    )

def run(in_path: str, out_sol: str, out_resid: str, time_col: str, cfg: WLSConfig):
    df = pd.read_csv(in_path)
    # Required columns check
    req = [
        "utcTimeMillis","Cn0DbHz","RawPseudorangeMeters",
        "SvPositionXEcefMeters","SvPositionYEcefMeters","SvPositionZEcefMeters",
        "SvElevationDegrees","SvAzimuthDegrees"
    ]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Optional columns
    if "TroposphericDelayMeters" not in df.columns: df["TroposphericDelayMeters"] = 0.0
    if "IonosphericDelayMeters" not in df.columns: df["IonosphericDelayMeters"] = 0.0
    if "SvClockBiasMeters" not in df.columns: df["SvClockBiasMeters"] = 0.0
    if "Svid" not in df.columns:
        df["Svid"] = df.get("SignalType", df.get("MessageType", "SV")).astype(str)

    key = time_col if time_col in df.columns else "utcTimeMillis"
    sols = []
    res_rows = []
    seed_xyz = None
    seed_cb = 0.0

    for t, rows in df.groupby(key):
        rows = rows.sort_values("Cn0DbHz", ascending=False)
        sol = solve_epoch_wls_from_rows(rows, seed_xyz, seed_cb, cfg)
        sols.append(sol)
        if sol.valid:
            seed_xyz = sol.ecef_xyz
            seed_cb = sol.clock_bias_m
            for r in sol.resid:
                res_rows.append({
                    "UnixTimeMillis": sol.unix_ms,
                    "sv": r.sv,
                    "el_deg": r.el_deg,
                    "az_deg": r.az_deg,
                    "cn0_dbhz": r.cn0_dbhz,
                    "weight": r.weight,
                    "residual_m": r.residual_m
                })

    sol_rows = []
    for s in sols:
        lat, lon, h = s.llh
        x, y, z = (s.ecef_xyz if isinstance(s.ecef_xyz, np.ndarray) else (np.nan, np.nan, np.nan))
        sol_rows.append({
            "UnixTimeMillis": s.unix_ms,
            "LatitudeDegrees": lat,
            "LongitudeDegrees": lon,
            "HeightMeters": h,
            "ecef_x": x, "ecef_y": y, "ecef_z": z,
            "clock_bias_m": s.clock_bias_m,
            "sigmaE": s.sigmaE, "sigmaN": s.sigmaN, "sigmaU": s.sigmaU,
            "nsats": s.nsats_used, "pdop": s.pdop, "valid": s.valid
        })
    pd.DataFrame(sol_rows).to_csv(out_sol, index=False)
    pd.DataFrame(res_rows).to_csv(out_resid, index=False)

def main():
    ap = argparse.ArgumentParser(description="Compute coarse WLS positions and residuals from device_gnss CSV.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input device_gnss CSV")
    ap.add_argument("--out", dest="out_sol", required=True, help="Output solutions CSV")
    ap.add_argument("--resid", dest="out_resid", required=True, help="Output residuals CSV")
    ap.add_argument("--time-col", dest="time_col", default="utcTimeMillis",
                    help="Epoch grouping column (default: utcTimeMillis). Alternatives: ArrivalTimeNanosSinceGpsEpoch, etc.")
    ap.add_argument("--el-min", type=float, default=12.0, help="Min elevation in degrees")
    ap.add_argument("--cn0-min", type=float, default=20.0, help="Min C/N0 in dB-Hz")
    ap.add_argument("--robust", dest="robust", action="store_true", help="Enable robust Huber reweighting")
    ap.add_argument("--no-robust", dest="robust", action="store_false", help="Disable robust reweighting")
    ap.set_defaults(robust=True)
    args = ap.parse_args()

    cfg = WLSConfig(el_min_deg=args.el_min, cn0_min_dbhz=args.cn0_min, robust=args.robust)
    run(args.in_path, args.out_sol, args.out_resid, args.time_col, cfg)

if __name__ == "__main__":
    main()
