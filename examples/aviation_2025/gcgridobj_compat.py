"""Import this before `aqhealth` / `gcgridobj`.

gcgridobj (2024) calls gcpy's old top-level names `csgrid_GMAO` and `make_grid_SG`; gcpy >= 1.5 renamed them
`csgrid_gmao` and `make_grid_sg`. This adds the old names as aliases. The cubed-sphere grid it then gives is
checked against GCHP output in test_gcgridobj_compat.py, because a mismatched grid would regrid wrongly
without raising an error.
"""
import gcpy

for old, new in (('csgrid_GMAO', 'csgrid_gmao'), ('make_grid_SG', 'make_grid_sg')):
    if not hasattr(gcpy, old):
        setattr(gcpy, old, getattr(gcpy, new))
