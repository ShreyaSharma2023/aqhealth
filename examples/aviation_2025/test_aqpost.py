"""Annual-mean MDA8 ozone from hourly ozone, in two definitions (aqpost.py).

Both are checked against slow reference implementations written independently of the vectorised code:
loops over explicit windows for the local-time definition, and pandas rolling windows (what plot_mda8.py
does) for the UTC one.
"""
import numpy as np
import pandas as pd
import pytest

import aqpost

HOURS_PER_DAY = 24


def hourly_ozone(rng, ndays, ncell):
    return 40.0 + 15.0 * rng.standard_normal((ndays * HOURS_PER_DAY, ncell))


# ---- reference implementations --------------------------------------------------------------


def reference_local(o3, lon, first_day, ndays, min_windows=18):
    """EPA-style: a local day's MDA8 is the highest 8-h mean among the windows starting in each hour of that
    local day (windows may run into the next day). Local time = UTC + round(lon / 15) hours."""
    out = np.full(o3.shape[1], np.nan)
    valid_days = np.zeros(o3.shape[1], dtype=int)
    for c in range(o3.shape[1]):
        offset = int(round(((lon[c] + 180.0) % 360.0 - 180.0) / 15.0))
        daily = []
        for d in range(ndays):
            start0 = HOURS_PER_DAY * (first_day + d) - offset
            means = [
                o3[s : s + 8, c].mean()
                for s in range(start0, start0 + HOURS_PER_DAY)
                if s >= 0 and s + 8 <= o3.shape[0]
            ]
            if len(means) >= min_windows:
                daily.append(max(means))
        valid_days[c] = len(daily)
        if daily:
            out[c] = np.mean(daily)
    return out, valid_days


def reference_utc(o3, first_day, ndays):
    """What plot_mda8.py does: trailing 8-h mean, daily maximum over UTC days."""
    index = pd.date_range('2025-01-01 00:00', periods=o3.shape[0], freq='h')
    out = []
    for c in range(o3.shape[1]):
        trailing = pd.Series(o3[:, c], index=index).rolling(8, min_periods=8).mean()
        daily = trailing.resample('1D').max().iloc[first_day : first_day + ndays]
        out.append(daily.mean())
    return np.array(out)


# ---- tests ----------------------------------------------------------------------------------


def test_local_offset_is_the_rounded_solar_hour():
    lon = np.array([0.0, 7.4, 7.6, 90.0, 179.9, 180.0, 200.0, 350.0, -7.6])
    assert aqpost.local_hour_offsets(lon).tolist() == [0, 0, 1, 6, 12, -12, -11, -1, -1]


def test_local_matches_reference_at_every_offset():
    rng = np.random.default_rng(1)
    first_day, ndays = 1, 10
    o3 = hourly_ozone(rng, first_day + ndays + 1, 27)
    lon = np.concatenate([15.0 * np.arange(-12, 13), [7.4, 7.6]])
    got, valid = aqpost.mda8_local(o3, lon, first_day, ndays)
    want, want_valid = reference_local(o3, lon, first_day, ndays)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    np.testing.assert_array_equal(valid, want_valid)


def test_utc_matches_pandas_rolling():
    rng = np.random.default_rng(2)
    first_day, ndays = 1, 10
    o3 = hourly_ozone(rng, first_day + ndays + 1, 6)
    got, valid = aqpost.mda8_utc(o3, first_day, ndays)
    np.testing.assert_allclose(got, reference_utc(o3, first_day, ndays), rtol=1e-12)
    assert (valid == ndays).all()


def test_constant_ozone_gives_that_ozone_in_both_definitions():
    o3 = np.full((5 * HOURS_PER_DAY, 4), 37.0)
    lon = np.array([0.0, 100.0, -120.0, 179.0])
    local, valid_local = aqpost.mda8_local(o3, lon, 1, 3)
    utc, valid_utc = aqpost.mda8_utc(o3, 1, 3)
    np.testing.assert_allclose(local, 37.0)
    np.testing.assert_allclose(utc, 37.0)
    assert (valid_local == 3).all() and (valid_utc == 3).all()


def test_periodic_ozone_gives_the_same_answer_whatever_the_day_boundary():
    """With the same daily cycle every day the highest 8-h mean does not depend on where the day starts, so
    both definitions, at any longitude, equal the highest 8-h mean of one cycle."""
    cycle = 30.0 + 25.0 * np.exp(-(((np.arange(24) - 15) / 3.0) ** 2))
    best = max(np.mean(np.roll(cycle, -s)[:8]) for s in range(24))
    o3 = np.tile(cycle, 8)[:, None] * np.ones((1, 5))
    lon = np.array([0.0, 60.0, 150.0, -75.0, -170.0])
    local, _ = aqpost.mda8_local(o3, lon, 1, 5)
    utc, _ = aqpost.mda8_utc(o3, 1, 5)
    np.testing.assert_allclose(local, best)
    np.testing.assert_allclose(utc, best)


def test_a_local_day_with_fewer_than_18_windows_is_left_out():
    """West of Greenwich the last local day needs hours that are after the end of the record. Fewer than 18 of
    its 24 windows exist, so that day is left out (the EPA completeness rule) and counted as not valid."""
    rng = np.random.default_rng(3)
    o3 = hourly_ozone(rng, 4, 2)  # days 0-3; analysis days 1, 2, 3
    lon = np.array([-120.0, 120.0])  # offsets -8 and +8
    got, valid = aqpost.mda8_local(o3, lon, 1, 3)
    assert valid.tolist() == [2, 3]
    want, _ = reference_local(o3, lon, 1, 3)
    np.testing.assert_allclose(got, want)


def test_nonfinite_ozone_is_an_error():
    o3 = np.full((3 * HOURS_PER_DAY, 2), 40.0)
    o3[10, 1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        aqpost.mda8_local(o3, np.array([0.0, 10.0]), 1, 1)
    with pytest.raises(ValueError, match='finite'):
        aqpost.mda8_utc(o3, 1, 1)


def test_the_record_must_cover_the_analysis_days():
    o3 = np.full((2 * HOURS_PER_DAY, 1), 40.0)
    with pytest.raises(ValueError, match='analysis days'):
        aqpost.mda8_utc(o3, 1, 5)
    with pytest.raises(ValueError, match='analysis days'):
        aqpost.mda8_local(o3, np.array([0.0]), 1, 5)
    with pytest.raises(ValueError, match='first_day'):
        aqpost.mda8_utc(o3, 0, 1)  # the trailing windows of the first day need the day before
