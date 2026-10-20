"""Annual surface concentrations of one GCHP run, as the health calculation reads them.

usage: aq-python postprocess_gchp.py RUN [--year 2025] [--runs-root ~/14.7.1] [--out-dir DIR]
       RUN is a run directory name under --runs-root (No_Aviation, Av_All, Av_NoWidebody, ...).

Reads  RUN/OutputDir/GEOSChem.AQHourly.YYYYMMDD_0000z.nc4        hourly surface O3 (mol/mol), one file per day
       RUN/OutputDir/GEOSChem.DefaultCollection.YYYYMMDD_0000z.nc4  daily PM25 (ug/m3, 72 levels) and Met_PMID
for the year, plus the last day of the year before (the hours local-time windows reach back into).
Writes DIR/RUN_YEAR_C90.nc on the C90 cubed sphere, dims (nf, Ydim, Xdim):
  O3_MDA8_ANN_LOCAL  ppbv   annual mean of the daily maximum 8-h ozone, local solar-time days (aqpost.mda8_local)
  O3_MDA8_ANN_UTC    ppbv   the same with UTC days and trailing windows, as plot_mda8.py (aqpost.mda8_utc)
  O3_ANN_MEAN        ppbv   plain annual mean ozone, for checking
  O3_VALID_DAYS_LOCAL days  days that entered O3_MDA8_ANN_LOCAL
  PM25_ANN           ug/m3  annual mean of the daily-mean PM25 at the surface (level 1)
The surface is level index 0 in these files (checked against Met_PMID); the lev attribute says 'down' but is not
used. A run whose year is not complete is an error.
"""
import argparse
import datetime as dt
import glob
import os
import subprocess

import netCDF4
import numpy as np

import aqpost

DEFAULT_RUNS_ROOT = '/home/shreya22/14.7.1'
DEFAULT_OUT_DIR = '/net/fs12/d1/shreya22/Health/concentrations'
CELL_CHUNK = 6000


def day_files(output_dir, collection, first, last):
    """Daily files of a collection for first..last inclusive; every day must exist."""
    files = []
    day = first
    while day <= last:
        path = os.path.join(output_dir, f'GEOSChem.{collection}.{day:%Y%m%d}_0000z.nc4')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        files.append((day, path))
        day += dt.timedelta(days=1)
    return files


def read_hourly_ozone(files):
    """(T, nf, Ydim, Xdim) ppbv, T = 24 per day, and the cell longitudes. Checks times, units and level."""
    chunks, lons = [], None
    for day, path in files:
        with netCDF4.Dataset(path) as ds:
            v = ds['SpeciesConcVV_O3']
            if v.units != 'mol mol-1 dry':
                raise ValueError(f'{path}: unexpected ozone units {v.units!r}')
            if v.shape[:2] != (24, 1):
                raise ValueError(f'{path}: expected 24 hourly records on one level, got {v.shape}')
            t = ds['time']
            if int(t.begin_date) != int(f'{day:%Y%m%d}') or int(t.time_increment) != 10000:
                raise ValueError(f'{path}: time axis is not hourly from {day}: {t.begin_date}, {t.time_increment}')
            chunks.append(np.asarray(v[:, 0], dtype=np.float32) * np.float32(1e9))
            if lons is None:
                lons = np.asarray(ds['lons'][:], dtype=np.float64)
    return np.concatenate(chunks), lons


def read_daily_pm25(files):
    """Mean of the daily-mean surface PM25 (nf, Ydim, Xdim) in ug/m3. Checks units and the level order."""
    total, n = None, 0
    for day, path in files:
        with netCDF4.Dataset(path) as ds:
            v = ds['PM25']
            if v.units != 'ug m-3' or v.shape[0] != 1 or v.shape[1] != 72:
                raise ValueError(f'{path}: unexpected PM25 {v.units!r} {v.shape}')
            pmid = ds['Met_PMID']
            if not (np.asarray(pmid[0, 0]).mean() > np.asarray(pmid[0, -1]).mean()):
                raise ValueError(f'{path}: level index 0 is not the surface (pressure does not fall with index)')
            field = np.asarray(v[0, 0], dtype=np.float64)
        total = field if total is None else total + field
        n += 1
    return total / n


def build(run_dir, year, out_path):
    output_dir = os.path.join(run_dir, 'OutputDir')
    first, last = dt.date(year, 1, 1), dt.date(year, 12, 31)
    hourly_files = day_files(output_dir, 'AQHourly', first - dt.timedelta(days=1), last)
    daily_files = day_files(output_dir, 'DefaultCollection', first, last)
    ndays = (last - first).days + 1

    o3, lon = read_hourly_ozone(hourly_files)
    nf, ny, nx = lon.shape
    flat = o3.reshape(o3.shape[0], -1)
    flat_lon = lon.reshape(-1)
    ncell = flat.shape[1]
    local, local_days = np.empty(ncell), np.empty(ncell, dtype=np.int16)
    utc, utc_days = np.empty(ncell), np.empty(ncell, dtype=np.int16)
    for lo in range(0, ncell, CELL_CHUNK):
        sl = slice(lo, min(lo + CELL_CHUNK, ncell))
        local[sl], local_days[sl] = aqpost.mda8_local(flat[:, sl], flat_lon[sl], 1, ndays)
        utc[sl], utc_days[sl] = aqpost.mda8_utc(flat[:, sl], 1, ndays)
    if (utc_days != ndays).any():
        raise RuntimeError('UTC-day MDA8 lost days: the hourly record has a gap')
    annual_mean = flat[24 : 24 * (ndays + 1)].astype(np.float64).mean(axis=0)
    pm25 = read_daily_pm25(daily_files)

    commit = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with netCDF4.Dataset(out_path, 'w') as out:
        out.createDimension('nf', nf), out.createDimension('Ydim', ny), out.createDimension('Xdim', nx)

        def put(name, data, units, long_name, dtype='f8'):
            v = out.createVariable(name, dtype, ('nf', 'Ydim', 'Xdim'), zlib=True, complevel=1)
            v[:] = np.asarray(data).reshape(nf, ny, nx)
            v.units, v.long_name = units, long_name

        with netCDF4.Dataset(hourly_files[0][1]) as first_file:
            lat = np.asarray(first_file['lats'][:], dtype=np.float64)
        put('lons', lon, 'degrees_east', 'cell centre longitude')
        put('lats', lat, 'degrees_north', 'cell centre latitude')
        put('O3_MDA8_ANN_LOCAL', local, 'ppbv', 'annual mean of daily max 8-h ozone, local solar-time days')
        put('O3_MDA8_ANN_UTC', utc, 'ppbv', 'annual mean of daily max trailing 8-h ozone, UTC days')
        put('O3_ANN_MEAN', annual_mean, 'ppbv', 'annual mean surface ozone')
        put('O3_VALID_DAYS_LOCAL', local_days, 'days', 'local days in O3_MDA8_ANN_LOCAL', dtype='i2')
        put('PM25_ANN', pm25, 'ug m-3', 'annual mean of daily-mean surface PM25')
        out.run, out.year = os.path.basename(os.path.normpath(run_dir)), year
        out.input_hourly_files = f'{len(hourly_files)} AQHourly files {hourly_files[0][0]}..{hourly_files[-1][0]}'
        out.input_daily_files = f'{len(daily_files)} DefaultCollection files {daily_files[0][0]}..{daily_files[-1][0]}'
        out.mda8_local = ('highest 8-h mean among windows starting in each hour of a local day (UTC + round(lon/15) h), '
                          'days with fewer than 18 windows left out')
        out.mda8_utc = 'trailing 8-h mean, highest per UTC day (plot_mda8.py)'
        out.created_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        out.aqhealth_commit = commit
    return dict(local=local.reshape(nf, ny, nx), utc=utc.reshape(nf, ny, nx), pm25=pm25, lat=None)


def summary(result, lats):
    w = np.cos(np.deg2rad(lats))
    mean = lambda x: float((x * w).sum() / w.sum())
    print(f'cos(lat)-weighted global mean MDA8: local {mean(result["local"]):.2f}  UTC {mean(result["utc"]):.2f} ppbv;'
          f'  PM25 {mean(result["pm25"]):.2f} ug/m3')
    d = result['local'] - result['utc']
    print(f'local - UTC MDA8: mean {mean(d):+.3f}, min {d.min():+.2f}, max {d.max():+.2f} ppbv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run')
    ap.add_argument('--year', type=int, default=2025)
    ap.add_argument('--runs-root', default=DEFAULT_RUNS_ROOT)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    run_dir = os.path.join(args.runs_root, args.run)
    out_path = os.path.join(args.out_dir, f'{args.run}_{args.year}_C90.nc')
    result = build(run_dir, args.year, out_path)
    with netCDF4.Dataset(out_path) as ds:
        summary(result, np.asarray(ds['lats'][:]))
    print('wrote', out_path)
