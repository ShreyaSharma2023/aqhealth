"""gcgridobj's C90 grid (what aqhealth regrids from) must be GCHP's C90 grid.

A different face orientation or longitude offset would regrid concentrations onto the wrong places without
any error. The reference is the coordinates GCHP wrote into its own output.
"""
import glob
import os

import numpy as np
import pytest
import xarray as xr

import gcgridobj_compat  # noqa: F401  (must come before gcgridobj)
from gcgridobj import cstools

GCHP_OUTPUT = os.environ.get('AQ_GCHP_SAMPLE_DIR', '/home/shreya22/14.7.1/No_Aviation/OutputDir')


@pytest.fixture(scope='module')
def gchp_sample():
    files = sorted(glob.glob(os.path.join(GCHP_OUTPUT, 'GEOSChem.AQHourly.2025*_0000z.nc4')))
    if not files:
        pytest.skip(f'no GCHP sample output in {GCHP_OUTPUT}')
    return xr.open_dataset(files[0])


def _wrapped_difference(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def test_cell_centres_match_gchp(gchp_sample):
    grid = cstools.gen_grid(90)
    assert grid['lat'].shape == gchp_sample['lats'].shape == (6, 90, 90)
    assert np.abs(grid['lat'].values - gchp_sample['lats'].values).max() < 1e-3
    assert _wrapped_difference(grid['lon'].values, gchp_sample['lons'].values).max() < 1e-3


def test_cell_corners_match_gchp(gchp_sample):
    grid = cstools.gen_grid(90)
    lat_gchp = gchp_sample['corner_lats'].values
    assert np.abs(grid['lat_b'].values - lat_gchp).max() < 1e-3
    dlon = _wrapped_difference(grid['lon_b'].values, gchp_sample['corner_lons'].values)
    # A corner exactly on a pole (C90 has one on the south pole) has no longitude: the two
    # grids may put it anywhere. Every other corner must agree.
    off_pole = np.abs(lat_gchp) < 90.0 - 1e-6
    assert dlon[off_pole].max() < 1e-3
    assert (~off_pole).sum() <= 2


def test_areas_add_up_to_the_sphere():
    grid = cstools.gen_grid(90)
    sphere = 4.0 * np.pi * 6.371e6**2
    assert float(grid['area'].sum()) == pytest.approx(sphere, rel=2e-3)
