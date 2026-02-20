HFCAD run notes
===============

Single-case run
---------------
python app/main.py -i app/input_test_1/h2000_M0.3_R1000km_PL1000kg.ini --outdir app/input_test_1


Multi-case sweep run (usual)
----------------------------
./run_sweep.sh --mode par -j 6 --in-dir app/input_ --out-dir results/input_

Common options:
- --mode seq|par
  seq = sequential, par = GNU parallel mode
- -j, --jobs N
  number of workers when --mode par
- --in-dir DIR
  directory containing *.ini case files
- --out-dir DIR
  root output directory passed to app/main.py
- --resume / --resume-from-out / --resume-run-dir / --resume-ok-list
  resume from previously completed cases
- --eta 0|1
  enable/disable live progress + ETA display
- --eta-interval SEC
  ETA refresh period in seconds


Automatic surrogate/cadquery subset for large sweeps
-----------------------------------------------------
run_sweep.sh can automatically split cases into:
- surrogate (FAST): calibrated compressor mass regression
- cadquery (SLOW): full CAD-based compressor mass model

Default policy:
- AUTO subset enabled
- activates when N_CASES >= 80
- cadquery fraction = 12% of cases
- cadquery min = 4, max = 32

Control flags:
- --auto-comp-subset 0|1
- --subset-threshold N
- --subset-fraction X      (0..1)
- --subset-min N
- --subset-max N

Example:
./run_sweep.sh --mode par -j 6 --in-dir app/input_sweep_r20 --out-dir results/input_sweep_r20 \
  --auto-comp-subset 1 --subset-threshold 120 --subset-fraction 0.08 --subset-min 4 --subset-max 20


INI option (per-case override baseline)
---------------------------------------
In each input INI [solver] section:

# Compressor mass model option:
# - surrogate: FAST (recommended for large sweeps), uses calibrated regression
# - cadquery: SLOW (higher fidelity), runs full CAD-based compressor mass model
compressor_mass_mode = surrogate

Note:
- run_sweep auto-subset sets per-case HFCAD_COMP_MASS_MODE env override at runtime.
- if auto-subset is disabled, solver.compressor_mass_mode in INI is used directly.


Requirements
------------
Create an environment and install dependencies:

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Optional extra packages for additional plotting/legacy scripts:
python -m pip install matplotlib scipy openpyxl plotly hickle streamlit cadquery pyvista stpyvista
