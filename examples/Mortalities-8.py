from aqhealth import mortality
import numpy as np
import os
from pathlib import Path
from gcgridobj import cstools
import xarray as xr
import pickle

# gpw_dir:      directory must contain the GPW 4.11 population grids
# who_dir:      directory must contain the GHE2016 CSVs
# mort_preproc: directory must contain GPW_to_WHO.pkl and (if GEMM used) gemm_crf.csv
# Paths are configurable via environment variables so this works across clusters.
HEALTH_ROOT = os.environ.get('AQHEALTH_HEALTH_ROOT', os.path.expanduser('~/HEALTH'))
DATA_ROOT = os.environ.get('AQHEALTH_DATA_ROOT', '/net/d16/data/shreya22')
COMMITTEE_DIR = os.environ.get('AQHEALTH_COMMITTEE_DIR', os.path.expanduser('~/committee_meeting_1/size_class/gchp_results'))

gpw_dir  = os.path.join(HEALTH_ROOT, 'GPW/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_totpop_2pt5_min_nc')
who_dir  = os.path.join(HEALTH_ROOT, 'Mortality_Files/LM_EXB_HealthDemo/WHO_Data')
mort_preproc = os.path.join(HEALTH_ROOT, 'Mortality_Files/LM_EXB_HealthDemo/Preprocessed_Mort_Data')
gemm_csv = os.path.join(mort_preproc, 'gemm_crf.csv')

data_dir = Path(COMMITTEE_DIR)
cs_res = 90


def load_field(path, var_names):
    with xr.open_dataset(path) as ds:
        var_name = next((name for name in var_names if name in ds.data_vars), None)
        if var_name is None:
            raise KeyError(f'{path} does not contain any of {var_names}; found {list(ds.data_vars)}')

        field = ds[var_name].squeeze(drop=True)
        if field.ndim != 3:
            raise ValueError(f'{path}:{var_name} should be 3-D after squeezing, got shape {field.shape}')

        return field.values


def load_scenario(name):
    o3 = load_field(data_dir / f'{name}_O3_Ann_MDA8.nc', ('SpeciesConcVV_O3', 'MDA8_O3'))
    pm25 = load_field(data_dir / f'{name}_PM25_Ann_Avg.nc', ('PM25',))
    return {'PM25': pm25, 'O3_MDA8_ANN': o3}


scenarios = {name: load_scenario(name) for name in ('NA', 'WB', 'NB', 'RJ')}

# WB/NB/RJ are absolute simulations with the NA background plus one size class.
# Reconstruct total aviation as NA plus each class perturbation.
pm_Av = (
    scenarios['NA']['PM25']
    + (scenarios['WB']['PM25'] - scenarios['NA']['PM25'])
    + (scenarios['NB']['PM25'] - scenarios['NA']['PM25'])
    + (scenarios['RJ']['PM25'] - scenarios['NA']['PM25'])
)
O3_Av = (
    scenarios['NA']['O3_MDA8_ANN']
    + (scenarios['WB']['O3_MDA8_ANN'] - scenarios['NA']['O3_MDA8_ANN'])
    + (scenarios['NB']['O3_MDA8_ANN'] - scenarios['NA']['O3_MDA8_ANN'])
    + (scenarios['RJ']['O3_MDA8_ANN'] - scenarios['NA']['O3_MDA8_ANN'])
)

cs_grid = cstools.gen_grid(cs_res)

# Each entry in chi_data corresponds to one simulation. For N simulations, we get
# N-1 results: the difference relative to ref_map.
chi_data = {
    'reference': {'PM25': pm_Av, 'O3_MDA8_ANN': O3_Av, 'grid': cs_grid},
    'Av': {'PM25': pm_Av, 'O3_MDA8_ANN': O3_Av, 'grid': cs_grid},
    'NA': {'PM25': scenarios['NA']['PM25'], 'O3_MDA8_ANN': scenarios['NA']['O3_MDA8_ANN'], 'grid': cs_grid},
    'WB': {'PM25': scenarios['WB']['PM25'], 'O3_MDA8_ANN': scenarios['WB']['O3_MDA8_ANN'], 'grid': cs_grid},
    'NB': {'PM25': scenarios['NB']['PM25'], 'O3_MDA8_ANN': scenarios['NB']['O3_MDA8_ANN'], 'grid': cs_grid},
    'RJ': {'PM25': scenarios['RJ']['PM25'], 'O3_MDA8_ANN': scenarios['RJ']['O3_MDA8_ANN'], 'grid': cs_grid},
}

# Which concentration set is the reference one?
ref_map = 'reference'

# What exposure response functions do we want to use?
# ...for PM2.5 use the GEMM set
ERF_set_full = mortality.gen_GEMM_erfs(gemm_csv)
# Only keep the NCD+LRI ERFs
ERF_set = [x for x in ERF_set_full if '_NCD+LRI_' in x.name]

# For ozone, append the ERF from Turner et al.
ERF_set.append(mortality.erf_lib['turner_2015_RD'])

# Number of samples (usually need at least 1000; use enough and check convergence)
n_samples = 2000

n_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

result_data = mortality.calc_mortality(
    chi_data,
    ref_map,
    common_chi_grid=True,
    ERF_set=ERF_set,
    GPW_dir=gpw_dir,
    WHO_dir=who_dir,
    n_rand=n_samples,
    n_cpus=n_cpus,
    mort_preprocessor_dir=mort_preproc,
)

# Save the output to a pickle file
output_file = 'test_size_class_avref.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(result_data, f)

print(f'Result data saved to {output_file}')

# Show results
for scen_name in chi_data.keys():
    if scen_name == ref_map:
        continue
    for erf in ERF_set:
        d_exp = np.nansum(result_data[erf.exposure_factor][erf.name][scen_name]['delta_exp'])
        aff_pop = np.nansum(result_data[erf.exposure_factor][erf.name]['affected_pop'])
        base_inc = np.nansum(result_data[erf.exposure_factor][erf.name]['base_morts'])
        print('ERF {:40s} due to {:15s}: {:9.0f} morts due to a change of {:12.0f} person-chi over {:12.0f} people (pop-exp: {:6.3f}, base inc: {:6.0f} x1,000)'.format(
            erf.name, scen_name,
            np.nansum(np.nanmean(result_data[erf.exposure_factor][erf.name][scen_name]['morts_vs_ref'], axis=-1)),
            d_exp, aff_pop, d_exp / aff_pop, base_inc / 1000.0))
