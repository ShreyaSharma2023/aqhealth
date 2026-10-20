"""Aviation air-quality mortality from the five 2025 GCHP runs.

usage: aq-python run_health_2025.py --mda8 {local,utc} [--o3-erf turner_2015_RD] [--n-samples 2000] [--tag NAME]

Concentrations come from postprocess_gchp.py (DIR/RUN_2025_C90.nc). The reference concentration, on which the
baseline incidence and the GEMM curve are evaluated, is the all-aviation run Av_All. Each other run is compared with
it:  morts_vs_ref = baseline incidence * (RR_scenario / RR_reference - 1).  For a run with aviation (or a class)
removed this is negative, so the deaths attributable to what was removed are  -morts_vs_ref:
    all aviation    = -(No_Aviation vs Av_All)
    wide-body       = -(Av_NoWidebody vs Av_All)       narrow-body, regional: likewise
Classes are not exactly additive (chemistry is non-linear): `class_sum_over_total` reports by how much.

Exposure-response: GEMM NCD+LRI (PM2.5, one function per age group) and one ozone function (default Turner 2015,
respiratory, age 30+, annual-mean MDA8). Population and baseline deaths are whatever aqhealth.mortality.calc_mortality
reads (GPW v4.11 2015 layer and WHO GHE2016): see its docstring. Writes DIR/health_TAG.pkl (full result) and
health_TAG_countries.csv / health_TAG_global.csv (mean and 95% interval over the Monte Carlo draws).
"""
import argparse
import os
import pickle

import netCDF4
import numpy as np
import pandas as pd

import gcgridobj_compat  # noqa: F401  (before gcgridobj / aqhealth)
from gcgridobj import cstools

from aqhealth import mortality

HEALTH_ROOT = os.environ.get('AQ_HEALTH_ROOT', '/net/fs12/d1/shreya22/Health')
GPW_DIR = os.path.join(HEALTH_ROOT, 'GPW/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_totpop_2pt5_min_nc')
DEMO_DIR = os.path.join(HEALTH_ROOT, 'LM_EXB_HealthDemo')
WHO_DIR = os.path.join(DEMO_DIR, 'WHO_Data')
PREPROC_DIR = os.path.join(DEMO_DIR, 'Preprocessed_Mort_Data')

def configure_regrid_cache():
    """Point gcgridobj's regridding-weight cache at shared storage and return the path.

    calc_mortality regrids C90 to the 2.5 arc-minute population grid with gcgridobj, which builds conservative weights
    (about 9 minutes per cube face, ~30 GB) and saves them in a relative directory unless told otherwise. Compute
    nodes cannot write under /home, and the weights are reused by every later run, so keep them with the data."""
    from gcgridobj import regrid

    path = os.environ.get('AQ_REGRID_DIR', os.path.join(HEALTH_ROOT, 'regridders'))
    os.makedirs(path, exist_ok=True)
    regrid.regrid_archive = path
    return path


REFERENCE_RUN = 'Av_All'
# scenario name -> (run, what was removed)
SCENARIOS = {'NA': 'No_Aviation', 'NoWB': 'Av_NoWidebody', 'NoNB': 'Av_NoNarrowbody', 'NoRJ': 'Av_NoRegional'}
CLASS_OF = {'NoWB': 'wide-body', 'NoNB': 'narrow-body', 'NoRJ': 'regional'}


def load_concentrations(conc_dir, mda8, year=2025):
    var = {'local': 'O3_MDA8_ANN_LOCAL', 'utc': 'O3_MDA8_ANN_UTC'}[mda8]
    grid = cstools.gen_grid(90)
    out = {}
    for name, run in [('reference', REFERENCE_RUN)] + list(SCENARIOS.items()):
        path = os.path.join(conc_dir, f'{run}_{year}_C90.nc')
        with netCDF4.Dataset(path) as ds:
            if int(ds.year) != year or ds.run != run:
                raise ValueError(f'{path} is for {ds.run} {ds.year}')
            o3, pm = np.asarray(ds[var][:], dtype=np.float64), np.asarray(ds['PM25_ANN'][:], dtype=np.float64)
        for x, label in ((o3, var), (pm, 'PM25_ANN')):
            if x.shape != (6, 90, 90) or not np.isfinite(x).all():
                raise ValueError(f'{path}: {label} is not a finite (6, 90, 90) field')
        out[name] = {'PM25': pm, 'O3_MDA8_ANN': o3, 'grid': grid}
    return out


def erf_set(o3_erf):
    erfs = [e for e in mortality.gen_GEMM_erfs(os.path.join(PREPROC_DIR, 'gemm_crf.csv')) if '_NCD+LRI_' in e.name]
    if o3_erf not in mortality.erf_lib or mortality.erf_lib[o3_erf].exposure_factor != 'O3_MDA8_ANN':
        raise ValueError(f'{o3_erf!r} is not an annual-mean MDA8 ozone function in aqhealth.mortality.erf_lib')
    return erfs + [mortality.erf_lib[o3_erf]]


def attributable(result, erfs, scenario):
    """Deaths attributable to what `scenario` removed: (n_country, n_draws) for PM2.5 and for ozone, summed over the
    ERFs of each pollutant within each draw."""
    pm = o3 = 0.0
    for erf in erfs:
        deaths = -np.nan_to_num(result[erf.exposure_factor][erf.name][scenario]['morts_vs_ref'])
        if erf.exposure_factor == 'PM25':
            pm = pm + deaths
        else:
            o3 = o3 + deaths
    return {'PM25': pm, 'O3': o3, 'total': pm + o3}


def summarise(result, erfs, countries):
    rows, glob = [], []
    totals = {}
    for scenario in SCENARIOS:
        att = attributable(result, erfs, scenario)
        for pollutant, deaths in att.items():
            lo, mid, hi = np.percentile(deaths.sum(axis=0), [2.5, 50, 97.5])
            glob.append(dict(scenario=scenario, pollutant=pollutant, mean=deaths.sum(axis=0).mean(), p2_5=lo, median=mid, p97_5=hi))
            totals[(scenario, pollutant)] = deaths.sum(axis=0)
            per_country = np.percentile(deaths, [2.5, 97.5], axis=1)
            rows += [dict(country=c, scenario=scenario, pollutant=pollutant, mean=deaths[i].mean(),
                          p2_5=per_country[0, i], p97_5=per_country[1, i]) for i, c in enumerate(countries)]
    glob_df = pd.DataFrame(glob)
    for pollutant in ('PM25', 'O3', 'total'):
        classes = sum(totals[(s, pollutant)] for s in CLASS_OF)
        ratio = classes / totals[('NA', pollutant)]
        glob_df = pd.concat([glob_df, pd.DataFrame([dict(
            scenario='class_sum_over_total', pollutant=pollutant, mean=ratio.mean(), p2_5=np.percentile(ratio, 2.5),
            median=np.median(ratio), p97_5=np.percentile(ratio, 97.5))])], ignore_index=True)
    return pd.DataFrame(rows), glob_df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mda8', choices=['local', 'utc'], required=True)
    ap.add_argument('--o3-erf', default='turner_2015_RD')
    ap.add_argument('--n-samples', type=int, default=2000)
    ap.add_argument('--conc-dir', default=os.path.join(HEALTH_ROOT, 'concentrations'))
    ap.add_argument('--out-dir', default=os.path.join(HEALTH_ROOT, 'results'))
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    tag = args.tag or f'{args.o3_erf}_{args.mda8}_n{args.n_samples}'
    os.makedirs(args.out_dir, exist_ok=True)

    print('regridding weights cached in', configure_regrid_cache())
    chi_data = load_concentrations(args.conc_dir, args.mda8)
    erfs = erf_set(args.o3_erf)
    result = mortality.calc_mortality(
        chi_data, 'reference', common_chi_grid=True, ERF_set=erfs, GPW_dir=GPW_DIR, WHO_dir=WHO_DIR,
        n_rand=args.n_samples, n_cpus=int(os.getenv('SLURM_CPUS_PER_TASK', 1)), mort_preprocessor_dir=PREPROC_DIR)
    with open(os.path.join(args.out_dir, f'health_{tag}.pkl'), 'wb') as f:
        pickle.dump(result, f)
    countries, glob_df = None, None
    countries = result['aux']['countries']
    country_df, glob_df = summarise(result, erfs, countries)
    country_df.to_csv(os.path.join(args.out_dir, f'health_{tag}_countries.csv'), index=False)
    glob_df.to_csv(os.path.join(args.out_dir, f'health_{tag}_global.csv'), index=False)
    print(glob_df.to_string(index=False, float_format=lambda x: f'{x:,.3f}' if abs(x) < 10 else f'{x:,.0f}'))
