# Repository Guidelines

## Project Structure & Module Organization
- `app/`: Primary Python sources, scripts, and notebooks for HFCAD sizing and analysis.
- `app/models/`: Reusable model components (stack, compressor, humidifier, heat exchanger).
- `app/old/`: Legacy and sensitivity scripts; keep changes minimal and well-scoped.
- `figs/`, `app/figs/`: Generated figures and plots.
- `results/`: Script outputs (tables, plots, intermediate artifacts).
- `media/`: Reference images and assets used in reports.
- Root files like `ReqPowDATA.xlsx` and `Power Mission Profile.png` are input data.

## Build, Test, and Development Commands
This repo is script-driven; there is no build system configured. Use the conda environment `aerofc`.

Common runs:
- `conda activate aerofc` — Activate the Python environment used for this project.
- `python app/HFCBattACDesign_SH_OOP_inpu_260117.py -i app/input_HFCAD_260126-2119.ini --outdir results/RIMP-03_260127-0440` — Main run command for the OOP sizing script and a standard input/output target.
  - The main script expects one INI input file and an output directory. It will create the output folder if it does not exist.
- `python app/gui_hfcad.py` — Launch PyQt GUI for sweep-case generation and execution.
  - Generates batch INI files from a template into a selected input folder, previews case differences, and runs cases sequentially/parallel with per-case logs under `results/_logs/run-YYYYMMDD-HHMMSS/`.

If you add new scripts, document the exact command and expected outputs here.

## Coding Style & Naming Conventions
- Python: 4-space indentation; prefer `snake_case` for functions/variables and `PascalCase` for classes.
- Keep numeric units explicit in names or comments (e.g., `_kg`, `_ft`, `_m`), as the code mixes SI/imperial.
- Prefer small, testable functions inside `app/models/` for new physics or sizing logic.

## Testing Guidelines
- No formal test framework is configured.
- For changes to sizing logic, add a small regression script or notebook cell that prints key outputs (mass, efficiency) and store plots in `results/` or `figs/` for review.
- When comparing, note input INI and major output metrics in your PR.

## Commit & Pull Request Guidelines
- Recent commit messages are short and generic (e.g., “Update”); there is no enforced convention.
- Prefer concise, imperative summaries (e.g., “Add compressor mass model checks”).
- PRs should include:
  - A brief description of the change and impacted scripts.
  - The exact command(s) run and input files used.
  - Output artifacts (plots or tables) in `results/`/`figs/` or a short summary of results.

## Data & Output Hygiene
- Treat large binaries (`.png`, `.xlsx`) as inputs or generated outputs; avoid committing regenerated artifacts unless needed for review.
- Use `results/` for new run outputs and keep `app/` focused on source code.
