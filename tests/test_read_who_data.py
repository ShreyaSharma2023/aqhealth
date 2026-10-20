"""read_who_data on the real GHE2016 files.

With pandas >= 2 (copy-on-write) a column's .values is read-only, so scaling it in place raised
'output array is read-only'. The function must not modify what pandas hands it.
"""
import os

import numpy as np
import pytest

from aqhealth import mortality

WHO_DIR = os.environ.get('AQ_WHO_DIR', '/net/fs12/d1/shreya22/Health/LM_EXB_HealthDemo/WHO_Data')
COUNTRY = os.path.join(WHO_DIR, 'GHE2016_Deaths_2016-country.xls')
REGION = os.path.join(WHO_DIR, 'GHE2016_Deaths_WHOReg_2000_2016.xls')

pytestmark = pytest.mark.skipif(not (os.path.exists(COUNTRY) and os.path.exists(REGION)),
                                reason='WHO GHE2016 files not available')

AGE_BANDS = ['0-4', '5-14', '15-29', '30-49', '50-59', '60-69', '70+']
REGIONS = ['Afr', 'Amr', 'Sear', 'Eur', 'Emr', 'Wpr', 'Global']


@pytest.fixture(scope='module')
def who_counts():
    return mortality.read_who_data(who_by_country_xls=COUNTRY, who_by_region_xls=REGION, as_rates=False)


def test_all_age_bands_and_regions_are_read(who_counts):
    assert list(who_counts) == AGE_BANDS
    for band in AGE_BANDS:
        assert set(REGIONS) <= set(who_counts[band].columns)


def test_region_columns_hold_finite_numbers(who_counts):
    for band in AGE_BANDS:
        for region in REGIONS:
            assert np.isfinite(np.asarray(who_counts[band][region], dtype=float)).all(), (band, region)


def test_the_regional_population_row_is_not_rescaled(who_counts):
    """Row 0 is the population; only the rows below it (deaths) are converted from thousands."""
    for region in REGIONS:
        total_deaths = sum(float(np.asarray(who_counts[b][region])[1]) for b in AGE_BANDS)
        population = sum(float(np.asarray(who_counts[b][region])[0]) for b in AGE_BANDS)
        assert population > 0 and 0 < total_deaths < population


def test_rates_are_deaths_per_person(who_counts):
    rates = mortality.read_who_data(who_by_country_xls=COUNTRY, who_by_region_xls=REGION, as_rates=True)
    glob_counts = np.asarray(who_counts['70+']['Global'], dtype=float)
    glob_rates = np.asarray(rates['70+']['Global'], dtype=float)
    np.testing.assert_allclose(glob_rates[1:], glob_counts[1:] / glob_counts[0], equal_nan=True)
