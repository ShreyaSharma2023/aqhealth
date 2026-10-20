"""Import this before `aqhealth` / `gcgridobj`.

gcgridobj (2024) predates two library changes, and each stops the health calculation:

1. It calls gcpy's old top-level names `csgrid_GMAO` and `make_grid_SG`; gcpy >= 1.5 renamed them `csgrid_gmao` and
   `make_grid_sg`. This adds the old names as aliases. The grid it then gives is checked against GCHP output in
   test_gcgridobj_compat.py, because a mismatched grid would regrid wrongly without raising an error.
2. Its cubed-sphere -> lat-lon regridding reads the output grid size from `Regridder.grid_out`, which xesmf >= 0.9
   no longer has. `c2l_arb` is replaced by the same function reading `shape_out` (n_lat, n_lon) instead.
"""
import gcpy
import numpy as np

for old, new in (('csgrid_GMAO', 'csgrid_gmao'), ('make_grid_SG', 'make_grid_sg')):
    if not hasattr(gcpy, old):
        setattr(gcpy, old, getattr(gcpy, new))

from gcgridobj import regrid  # noqa: E402  (after the gcpy aliases it needs)


def c2l_arb(cs_data, regridder_list):
    """Regrid cubed-sphere data (..., 6, N, N) to lat-lon (..., n_lat, n_lon): the six per-face regridders are
    applied and added. Leading dimensions are arbitrary. Same as gcgridobj.regrid.c2l_arb apart from the way the
    output size is found."""
    cs_data = np.asarray(cs_data)
    single_layer = cs_data.ndim == 3
    in_reshape = cs_data[np.newaxis] if single_layer else cs_data.reshape([-1] + list(cs_data.shape[-3:]))
    n_lat, n_lon = regridder_list[0].shape_out
    out = np.zeros((in_reshape.shape[0], n_lat, n_lon))
    for i_sample in range(in_reshape.shape[0]):
        for i_face in range(6):
            out[i_sample] += regridder_list[i_face](in_reshape[i_sample, i_face])
    return out[0] if single_layer else out.reshape(list(cs_data.shape[:-3]) + [n_lat, n_lon])


regrid.c2l_arb = c2l_arb
