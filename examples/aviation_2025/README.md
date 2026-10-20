# Aviation air-quality mortality, 2025 GCHP runs

Run everything with `./aq-python` (the venv `Health/venv-aqhealth` on the conda env `gchp_regrid`; it sets the
ESMF and PROJ variables conda would set on activation). Large jobs go to compute nodes; `edr` nodes are enough
for the post-processing and the tests.

1. `postprocess_gchp.py RUN` : annual surface concentrations of a GCHP run (MDA8 ozone in two definitions,
   PM2.5), written to `Health/concentrations/RUN_2025_C90.nc`. Needs the whole year, 2024-12-31 for the hourly
   files. Runs: No_Aviation, Av_All, Av_NoWidebody, Av_NoNarrowbody, Av_NoRegional.
2. `run_health_2025.py` : mortality with Av_All as the reference concentration (not yet run).

Tests (`./aq-python -m pytest examples/aviation_2025`): `aqpost.py` against slow reference implementations of both
MDA8 definitions; the C90 grid of gcgridobj against the coordinates GCHP writes (needs a GCHP output directory,
`AQ_GCHP_SAMPLE_DIR`).

`gcgridobj_compat.py` must be imported before `gcgridobj`: gcpy renamed two functions it calls.
