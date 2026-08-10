"""Command line NIRCam completeness calculator

Examples:
  # Mode 1: 50% completeness magnitude
  python3 completeness_cli.py --mode 1 --filter F090W --size 0.1 --background 0.3 --exposure 1000

  # Mode 2: completeness percentage at a given magnitude
  python3 completeness_cli.py --mode 2 --filter F090W --size 0.1 --background 0.3 --exposure 1000 --mag 26

  # Mode 2 without confusion (scaled to the plateau)
  python3 completeness_cli.py --mode 2 --filter F090W --size 0.1 --background 0.3 --exposure 1000 --mag 26 --no-confusion
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import special

FILTERS = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W']


def load_coefficients():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(script_dir, "coefficients.csv"))


def load_shape_params():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(script_dir, "shape_params.csv"))


def erfc_curve(x, d0, p, p1, p2, p3):
    return 0.5 * d0 * special.erfc(p * (x - p1)) * p2 + p3


def completeness_mag(filt, exposure_time, background, size, df):
    sub = df[df['filter'] == filt].sort_values('radius')
    radii = sub['radius'].values
    p = (sub['a'].values * (np.log10(exposure_time) - sub['e_avg'].values)
         + sub['b'].values * (background - sub['g_avg'].values)
         + sub['c_combined'].values)
    size = np.clip(size, radii.min(), radii.max())
    return np.interp(np.log10(size), np.log10(radii), p)


def completeness_mag_err(filt, exposure_time, background, size, df):
    sub = df[df['filter'] == filt].sort_values('radius')
    radii = sub['radius'].values
    coeff_var = (
        ((np.log10(exposure_time) - sub['e_avg'].values) ** 2) * sub['a_err'].values ** 2
        + ((background - sub['g_avg'].values) ** 2) * sub['b_err'].values ** 2
        + sub['c_combined_err'].values ** 2
    )
    total_sigma = np.sqrt(coeff_var + sub['rms'].values ** 2)
    size = np.clip(size, radii.min(), radii.max())
    return np.interp(np.log10(size), np.log10(radii), total_sigma)


def completeness_percent(filt, exposure_time, background, size, mag, coef_df, shape_df,
                         remove_confusion=False):
    p1 = completeness_mag(filt, exposure_time, background, size, coef_df)
    radii = shape_df['radius'].values
    lr = np.log10(np.clip(size, radii.min(), radii.max()))
    d0 = np.interp(lr, np.log10(radii), shape_df['d0'].values)
    p  = np.interp(lr, np.log10(radii), shape_df['p'].values)
    p2 = np.interp(lr, np.log10(radii), shape_df['p2'].values)
    p3 = np.interp(lr, np.log10(radii), shape_df['p3'].values)
    pct = erfc_curve(mag, d0, p, p1, p2, p3)
    if remove_confusion:
        plateau = erfc_curve(p1 - 10, d0, p, p1, p2, p3)
        if plateau > 0:
            pct = pct / plateau * 100.0
    return max(0.0, min(100.0, pct))


def completeness_percent_err(filt, exposure_time, background, size, mag, coef_df, shape_df):
    radii = shape_df['radius'].values
    lr = np.log10(np.clip(size, radii.min(), radii.max()))
    logr = np.log10(radii)
    p1 = completeness_mag(filt, exposure_time, background, size, coef_df)
    p1_err = completeness_mag_err(filt, exposure_time, background, size, coef_df)
    d0 = np.interp(lr, logr, shape_df['d0'].values)
    p  = np.interp(lr, logr, shape_df['p'].values)
    p2 = np.interp(lr, logr, shape_df['p2'].values)
    p3 = np.interp(lr, logr, shape_df['p3'].values)
    d0_s = np.interp(lr, logr, shape_df['d0_std'].values)
    p_s  = np.interp(lr, logr, shape_df['p_std'].values)
    p2_s = np.interp(lr, logr, shape_df['p2_std'].values)
    p3_s = np.interp(lr, logr, shape_df['p3_std'].values)
    rng = np.random.default_rng(0)
    n = 2000
    samples = erfc_curve(
        mag,
        rng.normal(d0, d0_s, n),
        rng.normal(p,  p_s,  n),
        rng.normal(p1, p1_err, n),
        rng.normal(p2, p2_s, n),
        rng.normal(p3, p3_s, n),
    )
    samples = np.clip(samples, 0.0, 100.0)
    return np.std(samples)


def main():
    parser = argparse.ArgumentParser(description="NIRCam completeness calculator (CLI)")
    parser.add_argument('--mode', type=int, choices=[1, 2], required=True,
                        help="1 = 50%% completeness magnitude, 2 = completeness percent at a magnitude")
    parser.add_argument('--filter', required=True, choices=FILTERS, dest='filt')
    parser.add_argument('--size', type=float, required=True, help="effective radius, arcsec (0.01 to 1.0)")
    parser.add_argument('--background', type=float, required=True, help="background, MJy/sr")
    parser.add_argument('--exposure', type=float, required=True, help="exposure time, seconds")
    parser.add_argument('--mag', type=float, help="object magnitude (required for mode 2)")
    parser.add_argument('--no-confusion', action='store_true',
                        help="mode 2 only: scale by the plateau to remove confusion")
    args = parser.parse_args()

    if not (0.01 <= args.size <= 1.0):
        parser.error("size must be between 0.01 and 1.0 arcsec")
    if args.exposure <= 0:
        parser.error("exposure must be positive")

    coef = load_coefficients()

    if args.mode == 1:
        mag = completeness_mag(args.filt, args.exposure, args.background, args.size, coef)
        err = completeness_mag_err(args.filt, args.exposure, args.background, args.size, coef)
        print(f"50% completeness magnitude: {mag:.2f} +/- {err:.2f}")
    else:
        if args.mag is None:
            parser.error("--mag is required for mode 2")
        shape = load_shape_params()
        pct = completeness_percent(args.filt, args.exposure, args.background, args.size,
                                   args.mag, coef, shape, remove_confusion=args.no_confusion)
        err = completeness_percent_err(args.filt, args.exposure, args.background, args.size,
                                       args.mag, coef, shape)
        tag = " (confusion removed)" if args.no_confusion else ""
        line = f"Completeness at mag {args.mag:g}{tag}: {pct:.1f}% +/- {err:.1f}%"
        if args.mag < 20 or args.mag > 32:
            line += f"\nNote: magnitude {args.mag:g} is outside the tested range of 20-32; extrapolated."
        print(line)


if __name__ == "__main__":
    main()