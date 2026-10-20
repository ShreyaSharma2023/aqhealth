"""Logic of run_health_2025.py that does not need the regridder or the population data.

`result` below has the structure aqhealth.mortality.calc_mortality returns:
result[exposure_factor][erf.name][scenario]['morts_vs_ref'] with shape (n_country, n_draws), negative for a scenario
with less pollution than the reference.
"""
import types

import numpy as np
import pytest

import run_health_2025 as rh

COUNTRIES = ['A', 'B']


def fake_erfs():
    pm_a = types.SimpleNamespace(name='GEMM_x_25-29', exposure_factor='PM25')
    pm_b = types.SimpleNamespace(name='GEMM_x_30-34', exposure_factor='PM25')
    o3 = types.SimpleNamespace(name='Turner', exposure_factor='O3_MDA8_ANN')
    return [pm_a, pm_b, o3]


def fake_result(erfs, removed):
    """removed[scenario] = (pm deaths per ERF, o3 deaths), the same in every draw, country 0 only."""
    result = {'PM25': {}, 'O3_MDA8_ANN': {}}
    for erf in erfs:
        result[erf.exposure_factor][erf.name] = {}
        for scenario, (pm, o3) in removed.items():
            deaths = pm if erf.exposure_factor == 'PM25' else o3
            morts = np.zeros((2, 4))
            morts[0] = -deaths  # fewer pollutants than the reference: negative
            result[erf.exposure_factor][erf.name][scenario] = {'morts_vs_ref': morts}
    return result


def test_attributable_deaths_are_minus_the_difference_to_the_reference():
    erfs = fake_erfs()
    result = fake_result(erfs, {'NA': (10.0, 5.0), 'NoWB': (6.0, 3.0), 'NoNB': (3.0, 2.0), 'NoRJ': (0.5, 0.1)})
    att = rh.attributable(result, erfs, 'NA')
    assert att['PM25'][0].tolist() == [20.0] * 4  # two PM2.5 functions of 10 each, summed within each draw
    assert att['O3'][0].tolist() == [5.0] * 4
    assert att['total'][0].tolist() == [25.0] * 4
    assert (att['total'][1] == 0).all()


def test_nan_in_a_cell_counts_as_no_deaths_not_as_a_nan_total():
    erfs = fake_erfs()
    result = fake_result(erfs, {'NA': (10.0, 5.0), 'NoWB': (6.0, 3.0), 'NoNB': (3.0, 2.0), 'NoRJ': (0.5, 0.1)})
    result['O3_MDA8_ANN']['Turner']['NA']['morts_vs_ref'][1] = np.nan
    assert np.isfinite(rh.attributable(result, erfs, 'NA')['total']).all()


def test_summary_reports_how_far_the_classes_are_from_adding_up_to_the_total():
    erfs = fake_erfs()
    removed = {'NA': (10.0, 5.0), 'NoWB': (6.0, 3.0), 'NoNB': (3.0, 2.0), 'NoRJ': (0.5, 0.1)}
    countries_df, global_df = rh.summarise(fake_result(erfs, removed), erfs, COUNTRIES)
    row = global_df[(global_df.scenario == 'class_sum_over_total') & (global_df.pollutant == 'total')].iloc[0]
    classes = (2 * 6.0 + 3.0) + (2 * 3.0 + 2.0) + (2 * 0.5 + 0.1)
    assert row['mean'] == pytest.approx(classes / (2 * 10.0 + 5.0))
    total = global_df[(global_df.scenario == 'NA') & (global_df.pollutant == 'total')].iloc[0]
    assert total['mean'] == pytest.approx(25.0)
    assert set(countries_df.country) == set(COUNTRIES)


def test_regridding_weights_are_cached_on_shared_storage(tmp_path, monkeypatch):
    """gcgridobj writes the weights it builds (about 9 minutes per cube face) to a relative directory unless told
    otherwise, and compute nodes cannot write under /home."""
    from gcgridobj import regrid

    monkeypatch.setenv('AQ_REGRID_DIR', str(tmp_path / 'weights'))
    path = rh.configure_regrid_cache()
    assert path == str(tmp_path / 'weights')
    assert (tmp_path / 'weights').is_dir()
    assert regrid.regrid_archive == path
