from aqhealth import mortality
import numpy as np
import os
from datetime import datetime, timedelta
from gcgridobj import cstools
import xarray as xr
import pickle

# gpw_dir:      directory must contain the GPW 4.11 population grids
# who_dir:      directory must contain the GHE2016 CSVs
# mort_preproc: directory must contain GPW_to_WHO.pkl and (if GEMM used) gemm_crf.csv
# Paths are configurable via environment variables so this works across clusters.
HEALTH_ROOT = os.environ.get('AQHEALTH_HEALTH_ROOT', os.path.expanduser('~/HEALTH'))
DATA_ROOT = os.environ.get('AQHEALTH_DATA_ROOT', '/net/d16/data/shreya22')
REFERENCE_DIR = os.environ.get('AQHEALTH_REFERENCE_DIR', '/net/d15/data/egibney/Sim_Data/Health_Impacts/Reference_2015')

gpw_dir  = os.path.join(HEALTH_ROOT, 'GPW/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_totpop_2pt5_min_nc')
who_dir  = os.path.join(HEALTH_ROOT, 'Mortality_Files/LM_EXB_HealthDemo/WHO_Data')
mort_preproc = os.path.join(HEALTH_ROOT, 'Mortality_Files/LM_EXB_HealthDemo/Preprocessed_Mort_Data')
gemm_csv = os.path.join(mort_preproc,'gemm_crf.csv')
who_by_country = os.path.join(who_dir,'GHE2016_Deaths_2016-country.xls')
who_by_region  = os.path.join(who_dir,'GHE2016_Deaths_WHOReg_2000_2016.xls')

# Load concentration data - Replace this with concentration data!
cs_res = 90
# Load in all the concentration data for each scenario
ds = xr.open_dataset(os.path.join(REFERENCE_DIR, 'Reference_2015_O3_Ann_MDA8.nc'))
O3_reference = ds['SpeciesConcVV_O3'][0,...].values
ds.close()
ds = xr.open_dataset(os.path.join(REFERENCE_DIR, 'Reference_2015_PM25_Ann_Avg.nc'))
pm_reference = ds['PM25'][0,0,...].values
ds.close()

ds = xr.open_dataset(os.path.join(DATA_ROOT, 'C90-NOAV/NOAV_O3_Ann_MDA8.nc'))
O3_noAv = ds['SpeciesConcVV_O3'][0,...].values
ds.close()
ds = xr.open_dataset(os.path.join(DATA_ROOT, 'C90-NOAV/NOAV_PM25_Ann_Avg.nc'))
pm_noAv = ds['PM25'][0,0,...].values
ds.close()

ds = xr.open_dataset(os.path.join(DATA_ROOT, 'C90-Glob19/Glob19_O3_Ann_MDA8.nc'))
O3_Av = ds['SpeciesConcVV_O3'][0,...].values
ds.close()
ds = xr.open_dataset(os.path.join(DATA_ROOT, 'C90-Glob19/Glob19_PM25_Ann_Avg.nc'))
pm_Av = ds['PM25'][0,0,...].values
ds.close()

cs_grid = cstools.gen_grid(cs_res)

# Each entry in chi_data corresponds to one simulation. For N simulations, we get N-1 results (difference relative to the reference simulation)
# Each simulation needs to have a grid specification
# Each simulation should have entries for every exposure factor, e.g.
# PM25: Annual mean PM2.5 (24h average), ug/m3
# O3_MDA8_ANN: Annuan mean MDA8 ozone (maximum daily average of 8-hour ozone, averaged over the year), ppbv
# O3_MDA1_SSN: Ozone-season mean (usually the 6 months during which that cell has the highest mean ozone) of the daily one-hour max ozone, ppbv
# Names for the exposure factors are dictated by the exposure_factor property of the ERFs in the mortality.py ERF class
# Here we have just two simulations (the minimum) and one exposure factor. We are also providing a 
# "reference" s
chi_data = {'reference': {'PM25': pm_Av, 'O3_MDA8_ANN': O3_Av, 'grid': cs_grid},
            'Av':        {'PM25': pm_Av, 'O3_MDA8_ANN': O3_Av, 'grid': cs_grid},
            'noAv':    {'PM25': pm_noAv, 'O3_MDA8_ANN': O3_noAv, 'grid': cs_grid},
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

# Number of samples (usually need at least 1000 - using 10 for demo, but
# for a real case you should go higher and check convergence)
n_samples = 2000

n_cpus = int(os.getenv('SLURM_CPUS_PER_TASK',1))       # Number of CPUs - hardcode if not in SLURM

result_data = mortality.calc_mortality(chi_data, ref_map, common_chi_grid=True, ERF_set=ERF_set, GPW_dir=gpw_dir, WHO_dir=who_dir, n_rand=n_samples, n_cpus=n_cpus, mort_preprocessor_dir=mort_preproc)

# Save the output to a pickle file
output_file = "turner.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(result_data, f)

print(f"Result data saved to {output_file}")

# Show results
for scen_name in chi_data.keys():
    if scen_name == ref_map:
        continue
    for erf in ERF_set:
        d_exp = np.nansum(result_data[erf.exposure_factor][erf.name][scen_name]['delta_exp'])
        aff_pop = np.nansum(result_data[erf.exposure_factor][erf.name]['affected_pop'])
        base_inc = np.nansum(result_data[erf.exposure_factor][erf.name]['base_morts'])
        print('ERF {:40s} due to {:15s}: {:9.0f} morts due to a change of {:12.0f} person-chi over {:12.0f} people (pop-exp: {:6.3f}, base inc: {:6.0f} x1,000)'.format(
            erf.name,scen_name,
            np.nansum(np.nanmean(result_data[erf.exposure_factor][erf.name][scen_name]['morts_vs_ref'],axis=-1)),
            d_exp,aff_pop,d_exp/aff_pop,base_inc/1000.0))

