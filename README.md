# NIRCam Completeness Calculator

A tool for estimating source completeness in JWST NIRCam imaging, based on the
SKYSURF-IR completeness analysis. It models completeness as a function of
filter, exposure time, background, and object effective radius.

There are two ways to use it: a graphical interface (`calc_gui.py`) and a
command line version (`calc_cli.py`). Both give the same results.

## What it does

- **50% completeness magnitude** — the magnitude at which half of the sources
  of a given size are recovered, for your chosen filter, exposure time, and
  background.
- **Completeness percentage** — the fraction of sources recovered at a specific
  magnitude, with an option to remove the confusion (field crowding) effect.

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/JWST_NIRCam_Completeness.git
cd JWST_NIRCam_Completeness
pip install -r requirements.txt
```

The GUI needs PyQt6. The command-line version does not. If you only want the
CLI, you can skip PyQt6 and install just numpy, pandas, and scipy.

## Files

- `calc_gui.py` — graphical calculator
- `calc_cli.py` — command-line calculator
- `coefficients.csv` — model coefficients per filter and radius
- `shape_params.csv` — completeness-curve shape parameters per radius
- `requirements.txt` — Python dependencies

Both `.csv` files must stay in the same folder as the scripts.

## Graphical use

```bash
python3 calc_gui.py
```

Tab 1 returns the 50% completeness magnitude. Tab 2 returns the completeness
percentage at a magnitude you enter, with a toggle for including or removing
confusion.

## Command-line use

Mode 1 — 50% completeness magnitude:

```bash
python3 calc_cli.py --mode 1 --filter F090W --size 0.1 --background 0.3 --exposure 1000
```

Mode 2 — completeness percentage at a magnitude:

```bash
python3 calc_cli.py --mode 2 --filter F090W --size 0.1 --background 0.3 --exposure 1000 --mag 26
```

Add `--no-confusion` to scale by the plateau and report detection completeness
with confusion removed:

```bash
python3 calc_cli.py --mode 2 --filter F090W --size 0.1 --background 0.3 --exposure 1000 --mag 26 --no-confusion
```

For a full list of options:

```bash
python3 calc_cli.py --help
```

## Inputs

- **filter** — one of F090W, F115W, F150W, F200W, F277W, F356W, F410M, F444W
- **size** — object effective radius in arcseconds, from 0.01 to 1.0
- **background** — sky background in MJy/sr
- **exposure** — exposure time in seconds
- **mag** — object AB magnitude (mode 2 only); tested range is 20 to 32

## Notes

- Results are calibrated on the SKYSURF-IR sample and are most reliable within
  the tested magnitude range of 20 to 32. Values outside this range are
  extrapolated and flagged as such.
- The "with confusion" percentage reflects completeness as actually observed,
  where the bright-end plateau falls below 100% because injected sources can
  land on real objects. The "without confusion" option divides by that plateau
  to isolate detection against noise alone.

## Citation

If you use this tool, please cite the SKYSURF-IR completeness paper (Oskouie et
al.).
