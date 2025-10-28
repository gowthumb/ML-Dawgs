#!/usr/bin/env python3
# ppk_batch_v5.py
import argparse, os, re, sys, gzip, shutil, time, signal, hashlib
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError
from tqdm import tqdm
import ssl, certifi


# Optional libs
try:
    import requests
except Exception:
    requests = None
try:
    from hatanaka import RINEXFile
    HAS_HATANAKA = True
except Exception:
    HAS_HATANAKA = False

def log(s): print(f"[ppk] {s}")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True); return p

def parse_date_from_dataset_path(p: Path) -> datetime:
    # .../YYYY-MM-DD-COUNTRY-CITY-#/...
    s = str(p).replace("\\","/")
    m = re.search(r"/(\d{4})-(\d{2})-(\d{2})-", s)
    if not m: raise ValueError(f"Date not found in path: {p}")
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)

def doy(dt: datetime) -> int:
    return int(dt.strftime("%j"))
def yy(dt: datetime)  -> str:
    return dt.strftime("%y")

def gunzip_if_needed(p: Path) -> Path:
    if p.suffix.lower() == ".gz":
        out = p.with_suffix("")
        with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        try: p.unlink()
        except: pass
        return out
    return p

def uncompress_z_if_needed(p: Path) -> Path:
    """Decompress .Z files using Unix compress format or gzip"""
    if p.suffix.upper() == ".Z":
        out = p.with_suffix("")
        try:
            import subprocess
            # Try using system uncompress command (Windows may not have this)
            result = subprocess.run(['uncompress', str(p)], capture_output=True)
            if result.returncode == 0 and out.exists():
                return out
        except:
            pass
        
        # Fallback: try gzip (sometimes .Z files are actually gzipped)
        try:
            with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            try: p.unlink()
            except: pass
            return out
        except:
            # If both fail, try reading as raw and write out
            # This handles the Unix compress format manually
            try:
                import unlzw3
                with open(p, "rb") as fi:
                    compressed_data = fi.read()
                    decompressed = unlzw3.unlzw(compressed_data)
                with open(out, "wb") as fo:
                    fo.write(decompressed)
                try: p.unlink()
                except: pass
                return out
            except Exception as e:
                log(f"Failed to decompress {p.name}: {e}")
                # Return the .Z file as-is; rnx2rtkp might handle it
                return p
    return p

def convert_to_rnx_if_needed(p: Path) -> Path:
    # Un-gzip first, then convert Hatanaka (.crx or *.yyd) to plain RINEX
    p = gunzip_if_needed(p)
    name = p.name.lower()
    is_hatanaka = name.endswith(".crx") or re.search(r"\.\d{2}d$", name) is not None
    if is_hatanaka and HAS_HATANAKA:
        out = p.with_suffix(".rnx")
        try:
            RINEXFile(p).to_rinex(out)
            return out
        except Exception as e:
            log(f"hatanaka conversion failed for {p.name}: {e}")
    return p

# ---------------- NAV: reuse from train, else try BKG/IGN ----------------
def find_brdc(train_root: Path, dt: datetime):
    pat = f"brdc{doy(dt):03d}0.{yy(dt)}n"
    for p in train_root.rglob(pat): return p
    for p in train_root.rglob(pat + "*"): return p
    return None

def brdc_candidate_urls(dt: datetime):
    Y, D, y = dt.year, doy(dt), yy(dt)
    return [
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{Y}/{D:03d}/brdc{D:03d}0.{y}n.Z",
        f"https://igs.ign.fr/IGS/BRDC/{Y}/{D:03d}/brdc{D:03d}0.{y}n.Z",
        # CDDIS often requires auth; try last:
        f"https://cddis.nasa.gov/archive/gnss/data/daily/{Y}/{D:03d}/{y}n/brdc{D:03d}0.{y}n.Z",
    ]

# ---------------- Base RINEX URL builders ----------------
def s3_daily_d(code, dt):  # NOAA CORS S3 daily decimated (most common)
    D, y, s = doy(dt), yy(dt), code.lower()
    return f"https://noaa-cors-pds.s3.amazonaws.com/rinex/{dt.year}/{D:03d}/{s}/{s}{D:03d}0.{y}d.gz"

def s3_daily_o(code, dt):  # NOAA CORS S3 daily obs (sometimes present)
    D, y, s = doy(dt), yy(dt), code.lower()
    return f"https://noaa-cors-pds.s3.amazonaws.com/rinex/{dt.year}/{D:03d}/{s}/{s}{D:03d}0.{y}o.gz"

def gage_daily_d(code, dt):  # EarthScope/GAGE (needs EARTHSCOPE_TOKEN)
    D, y, s = doy(dt), yy(dt), code.lower()
    return f"https://gage-data.earthscope.org/archive/gnss/rinex/obs/{dt.year}/{D:03d}/{s}{D:03d}0.{y}d.Z"

def gage_daily_o(code, dt):
    D, y, s = doy(dt), yy(dt), code.lower()
    return f"https://gage-data.earthscope.org/archive/gnss/rinex/obs/{dt.year}/{D:03d}/{s}{D:03d}0.{y}o.Z"

def ngs_rnx3_crx(code, dt):  # Rare in 2020; try last
    Y, D, s = dt.year, doy(dt), code.lower()
    return f"https://geodesy.noaa.gov/corsdata/rinex3/{Y}/{D:03d}/{s}/{s}_R_{Y}{D:03d}0000_01D_30S_MO.crx.gz"

# ---------------- Download helpers ----------------
def _req(url, timeout=45): return Request(url, headers={"User-Agent":"ppk-batch/5.0","Accept":"/"})
def download_stream(url, dest: Path, timeout=60, retries=2) -> bool:
    for i in range(retries):
        try:
            # Prefer requests (verifies with certifi)
            if requests:
                with requests.get(url, stream=True, timeout=timeout, verify=certifi.where()) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    total = int(r.headers.get("Content-Length") or 0)
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f, tqdm(total=total if total>0 else None,
                            unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
                        for chunk in r.iter_content(1024*64):
                            if not chunk: continue
                            f.write(chunk)
                            if total: bar.update(len(chunk))
                    tmp.replace(dest)
                    return True
            # Fallback: urllib with certifi-backed SSL context
            ctx = ssl.create_default_context(cafile=certifi.where())
            with urlopen(Request(url, headers={"User-Agent":"ppk-batch/5.0","Accept":"/"}), timeout=timeout, context=ctx) as r:
                total = int(r.headers.get("Content-Length") or 0)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f, tqdm(total=total if total>0 else None,
                        unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
                    while True:
                        chunk = r.read(1024*64)
                        if not chunk: break
                        f.write(chunk)
                        if total: bar.update(len(chunk))
                tmp.replace(dest)
                return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if i == retries-1:
                log(f"download failed: {url} ({e})")
                return False
            time.sleep(1.5*(i+1))
    return False


def try_gage(url, dest: Path) -> bool:
    token = os.getenv("EARTHSCOPE_TOKEN")
    if not token or not requests: return False
    try:
        with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=90) as r:
            if r.status_code != 200: return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = int(r.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as f, tqdm(total=total if total>0 else None,
                    unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
                for chunk in r.iter_content(1024*64):
                    if chunk: f.write(chunk); 
                    if total: bar.update(len(chunk))
            tmp.replace(dest); return True
    except KeyboardInterrupt: raise
    except Exception as e:
        log(f"GAGE download failed: {e}")
        return False

# ---------------- Per-dataset processing ----------------
def choose_station_sequence(ds_dir: Path, args):
    # Rotate list per dataset for load-spread
    seq = [s.strip().upper() for s in args.station_list.split(",") if s.strip()]
    if not seq: raise ValueError("Empty --station-list")
    h = int(hashlib.md5(ds_dir.name.encode("utf-8")).hexdigest(), 16)
    k = h % len(seq)
    return seq[k:] + seq[:k]

def try_download_base(stations, dt, outdir: Path):
    for st in stations:
        # Best-first order for 2020 era:
        candidates = [
            s3_daily_d(st, dt),
            s3_daily_o(st, dt),
            gage_daily_d(st, dt),
            gage_daily_o(st, dt),
            ngs_rnx3_crx(st, dt),
        ]
        for u in candidates:
            dest = outdir / Path(u).name
            if ("gage-data.earthscope.org" in u):
                if try_gage(u, dest): return dest
            else:
                if download_stream(u, dest): return dest
    return None

def brdc_fetch_if_needed(train_root: Path, dt: datetime, outdir: Path):
    nav = find_brdc(train_root, dt)
    if nav:
        nav = gunzip_if_needed(nav) if nav.suffix.lower()==".gz" else nav
        return uncompress_z_if_needed(nav)
    for u in brdc_candidate_urls(dt):
        dest = outdir / Path(u).name
        if download_stream(u, dest): return uncompress_z_if_needed(dest)
    log("WARN: BRDC nav not obtained (often already in train/_nav)"); return None

def process_one(rover_path: Path, train_root: Path, out_root: Path, args):
    # Find the phone folder (parent of rover file or parent of supplemental)
    if rover_path.parent.name == "supplemental":
        phone_dir = rover_path.parents[1]
        date_dir = rover_path.parents[2]
    else:
        phone_dir = rover_path.parent
        date_dir = rover_path.parents[1]
    
    # Create unique output folder name combining date and phone
    ds_name = f"{date_dir.name}-{phone_dir.name}"
    dt = parse_date_from_dataset_path(date_dir)  # Get date from date folder
    outdir = ensure_dir(out_root / ds_name)

    # Rover
    rover_out = outdir / rover_path.name
    if rover_out.resolve() != rover_path.resolve():
        shutil.copy2(rover_path, rover_out)

    # Nav
    nav_local = brdc_fetch_if_needed(train_root, dt, outdir)

    # Base - use phone_dir for station rotation
    seq = choose_station_sequence(phone_dir, args)
    base = try_download_base(seq, dt, outdir)
    if not base:
        raise RuntimeError(f"[{ds_name}] base download failed for stations: {','.join(seq)}")
    base_use = convert_to_rnx_if_needed(base)

    return {"dataset": ds_name, "rover": rover_out, "base": base_use, "nav": nav_local}

'''
This will create outputs like:
ppk_output_z/
  2020-05-14-GooglePixel4/
  2020-05-14-GooglePixel4XL/
  2020-05-14-SamsungGalaxyS20/ '''

def main():
    ap = argparse.ArgumentParser(description="Batch fetch PPK inputs for all datasets in train/")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--station-list", required=True, help="Comma-separated CORS codes, e.g., P224,P225,P222,P231,P277")
    args = ap.parse_args()

    train_root = Path(args.train_root).expanduser().resolve()
    out_root   = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rovers = sorted(train_root.glob("**/supplemental/gnss_rinex.*o"))
    if not rovers: sys.exit(f"No rover files found under {train_root}")

    # Ctrl-C friendly
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    ok, fail = 0, 0
    with tqdm(total=len(rovers), desc="Datasets", unit="set") as pbar:
        for r in rovers:
            if stop["flag"]: break
            try:
                process_one(r, train_root, out_root, args)
                ok += 1
            except Exception as e:
                log(str(e)); fail += 1
            pbar.update(1)
    log(f"Done. success={ok}, failed={fail}")
    if fail:
        log("Tip: add more station codes to --station-list or verify those stations had data on those dates.")

if __name__ == "__main__":
    main()


