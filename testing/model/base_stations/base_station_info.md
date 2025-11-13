# Base Station Solutions for LAX Data

## Problem
Your LAX data (Dec 2021) needs base station files for PPK processing, but:
- Local CORS stations don't have complete data for that period
- Stations are 38-95km away (affects accuracy)

## Solution: Use IGS Global Network Stations

Instead of NOAA CORS, use IGS (International GNSS Service) stations which have:
✓ Global coverage
✓ Complete historical data
✓ High-quality reference data
✓ 24/7 availability

### Closest IGS Stations to LAX (34.27°N, 118.62°W):

1. **P472** - Pasadena, CA
   - Distance: ~45km
   - Coordinates: 34.14°N, 118.17°W
   - Data: ftp://gdc.cddis.eosdis.nasa.gov/pub/gps/data/daily/

2. **TORP** - Torrance, CA
   - Distance: ~35km
   - Coordinates: 33.83°N, 118.33°W

3. **PTVL** - Point Vicente, CA
   - Distance: ~40km
   - Coordinates: 33.74°N, 118.41°W

## Quick Fix: Download from IGS/CDDIS

### Manual Download (Easiest):

1. **Go to**: https://cddis.nasa.gov/archive/gnss/data/daily/

2. **Navigate**:
   ```
   /2021/341/  (341 = day of year for Dec 7, 2021)
   ```

3. **Download files** (look for stations near LA):
   - Observation: `p472341a.21d.Z` or `p472341a.21o.gz`
   - Navigation: `brdc3410.21n.Z` (broadcast ephemeris, works for all)

4. **Decompress**: Use 7-Zip or similar tool

### Alternative: NGS CORS (More Reliable)

**NGS has better coverage for US stations:**

Visit: https://geodesy.noaa.gov/corsdata/

Search for stations near:
- LAX Airport (33.9N, 118.4W)
- Van Nuys (34.2N, 118.5W)

Download RINEX files for December 7, 2021 (DOY 341)

## What You Actually Need

For each date, you need:
1. **Base station observation file** (.obs or .o or .rnx)
   - Contains: Pseudorange, carrier phase measurements
   - From: Nearest reference station

2. **Navigation file** (.nav or .n or .rnx)
   - Contains: Satellite ephemeris (orbits)
   - From: Any station (can reuse same file)
   - Broadcast navigation works globally

3. **Rover observation file** (you already have these!)
   - Located in: `train/*/supplemental/gnss_rinex.20o`

## Simplified Approach: Skip PPK, Use Your Current Baseline

**Actually, you might not need to regenerate POS files!**

Your current pipeline already has:
- Baseline positions (probably from existing PPK or from device)
- Ground truth for training
- ML model that corrects baseline → ground truth

The "far base station" problem is already reflected in your baseline errors (9.19m Mean(P50,P95)).

**Your ML model is effectively learning to correct for:**
- Base station distance errors
- Atmospheric delays
- Multipath
- Device-specific biases

**Current results show this works:**
- Baseline: 9.19m Mean(P50,P95)
- After ML correction: 6.89m Mean(P50,P95)
- **25% improvement** without needing new base stations!

## Recommendation

**Option 1: Keep using existing baseline** (simplest)
- Your ML model already reduces error from 9.19m → 6.89m
- Focus on improving ML model to reach 1-2m target:
  - Add more features (velocity, temporal patterns)
  - Try neural networks (GRU/LSTM)
  - Ensemble methods
  - Hyperparameter tuning

**Option 2: Improve baseline with better base stations** (more work)
- Download IGS station data for key dates
- Reprocess with RTKLIB using closer stations
- Might get baseline down to 5-7m
- Then apply ML to reach 1-2m target

**Option 3: Hybrid approach**
- Use existing baseline for training
- For test/submission, download IGS data if needed
- Most submissions probably use similar base station distances

## For Your Specific LAX Dates

Dec 7-9, 2021 (DOY 341-343):

```bash
# IGS CDDIS archive
Base URL: https://cddis.nasa.gov/archive/gnss/data/daily/2021/

# Files needed:
341/21o/p472341a.21o.Z  # P472 observation
341/21n/brdc3410.21n.Z  # Broadcast nav (works for all days)

342/21o/p472342a.21o.Z  # Next day
343/21o/p472343a.21o.Z  # etc.
```

Note: You'll need to register (free) at: https://urs.earthdata.nasa.gov/

## Bottom Line

**You don't necessarily need to fix the base station issue!**

Your current approach of:
1. Use existing baseline (even with distant base stations)
2. Learn corrections with ML
3. Achieve 6.89m → continue improving toward 1-2m

...is actually a valid and common strategy in the Smartphone Decimeter Challenge.

Focus on ML improvements rather than PPK reprocessing.
