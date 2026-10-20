"""Annual-mean MDA8 ozone from hourly ozone, in two definitions.

Input is an hourly record `o3` of shape (T, ncell) in ppbv whose hour 0 is 00:00 UTC of day 0; the analysis
covers `ndays` days starting at day `first_day` (the days before and after the analysis days supply the hours the
windows reach into). Cells are independent, so any grid can be flattened into `ncell`.

- `mda8_local`: the regulatory form. For each local day, the highest 8-h mean among the windows that START in
  each hour of that day (a window may run into the next day). Local time is UTC + round(longitude / 15) hours,
  not the political time zone. A day with fewer than `min_windows` (18) usable windows is left out, as the EPA
  completeness rule does; at the end of the record this only affects the last day west of Greenwich.
- `mda8_utc`: what plot_mda8.py does. Trailing 8-h mean (the window ends at the hour), highest per UTC day.

Both return the mean over the valid days and the number of valid days per cell.
"""
import numpy as np

HOURS_PER_DAY = 24
WINDOW_HOURS = 8


def local_hour_offsets(lon_deg):
    """Whole hours to add to UTC for the local solar time of each longitude (degrees east)."""
    lon = (np.asarray(lon_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return np.rint(lon / 15.0).astype(int)


def _check(o3, first_day, ndays):
    o3 = np.asarray(o3)
    if o3.ndim != 2:
        raise ValueError(f'o3 must be (hours, cells), got shape {o3.shape}')
    if not np.isfinite(o3).all():
        raise ValueError('ozone record must be finite: a missing hour would silently shorten windows')
    if first_day < 0 or ndays < 1 or HOURS_PER_DAY * (first_day + ndays) > o3.shape[0]:
        raise ValueError(
            f'the record has {o3.shape[0]} hours, too short for {ndays} analysis days from day {first_day}'
        )
    return o3


def _forward_means(x):
    """8-h mean of x[t : t + 8] for every t, NaN where the window runs past the end. x is (T, n)."""
    csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)])
    means = np.full(x.shape, np.nan)
    means[: x.shape[0] - WINDOW_HOURS + 1] = (csum[WINDOW_HOURS:] - csum[:-WINDOW_HOURS]) / WINDOW_HOURS
    return means


def _mean_of_daily_maxima(windows, min_windows):
    """windows: (ndays, 24, n) of 8-h means, NaN where a window does not exist."""
    count = np.isfinite(windows).sum(axis=1)
    with np.errstate(all='ignore'):
        daily = np.nanmax(np.where(np.isfinite(windows), windows, -np.inf), axis=1)
    daily = np.where(count >= min_windows, daily, np.nan)
    valid = np.isfinite(daily).sum(axis=0)
    with np.errstate(all='ignore'):
        mean = np.where(valid > 0, np.nansum(daily, axis=0) / np.maximum(valid, 1), np.nan)
    return mean, valid


def mda8_local(o3, lon, first_day, ndays, min_windows=18):
    o3 = _check(o3, first_day, ndays)
    offsets = local_hour_offsets(lon)
    if offsets.shape != (o3.shape[1],):
        raise ValueError('lon must have one value per cell')
    mean = np.full(o3.shape[1], np.nan)
    valid = np.zeros(o3.shape[1], dtype=int)
    day = np.arange(ndays)[:, None] + first_day
    for offset in np.unique(offsets):
        cells = np.flatnonzero(offsets == offset)
        forward = _forward_means(np.asarray(o3[:, cells], dtype=np.float64))
        # UTC hour at which the window starting in local hour h of local day d begins.
        start = HOURS_PER_DAY * day - offset + np.arange(HOURS_PER_DAY)[None, :]
        inside = (start >= 0) & (start < o3.shape[0])
        windows = np.where(inside[:, :, None], forward[np.clip(start, 0, o3.shape[0] - 1)], np.nan)
        mean[cells], valid[cells] = _mean_of_daily_maxima(windows, min_windows)
    return mean, valid


def mda8_utc(o3, first_day, ndays, min_windows=18):
    if first_day < 1:
        raise ValueError('first_day must be at least 1: the trailing windows of the first day need the day before')
    o3 = _check(o3, first_day, ndays)
    forward = _forward_means(np.asarray(o3, dtype=np.float64))
    # The trailing window ending at hour e is the forward window starting at e - 7.
    end = HOURS_PER_DAY * (np.arange(ndays)[:, None] + first_day) + np.arange(HOURS_PER_DAY)[None, :]
    windows = forward[end - (WINDOW_HOURS - 1)]
    return _mean_of_daily_maxima(windows, min_windows)
