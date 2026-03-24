"""
Doppler-Broadened Two-Photon Absorption Spectroscopy Analysis
=============================================================
Data columns: power (mW) | wavelength (nm) | PMT singles (counts) |
              frequency (THz) | photodiode voltage (V)

Doppler-broadened profiles are fit with a Gaussian (thermal velocity
distribution). FWHM ∝ √T gives a temperature diagnostic independent of
the thermocouple reading.

Usage:
    python doppler_broadened_analysis.py
    python doppler_broadened_analysis.py --data_dir /path/to/data --output_dir /path/to/output
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import glob, re, os, argparse

# ── constants ──────────────────────────────────────────────────────────────
C_LIGHT   = 2.99792458e8          # m/s
LAMBDA_0  = 822.468e-9            # m  (nominal TPA wavelength)
M_CS      = 132.905e-3 / 6.022e23 # kg  (Cs-133)
KB        = 1.380649e-23           # J/K

# ── style ──────────────────────────────────────────────────────────────────
BG_COLOR     = "white"
PANEL_BG     = "white"
TEXT_COLOR   = "black"
ACCENT_COLOR = "#1f77b4"

FILE_PATTERN = "DopplerBroadened*C*.txt"

COLUMNS = {"power": 0, "wavelength": 1, "pmt": 2, "frequency": 3, "voltage": 4}


# ── helpers ────────────────────────────────────────────────────────────────

def voltage_to_intensity(voltage_V):
    """Convert photodiode voltage (V) to intensity (mW) via y = 2.133x - 0.05."""
    return 2.133 * voltage_V - 0.05


def set_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.spines[:].set_color("#aaaaaa")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # Grids removed
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)


def gaussian(x, amplitude, center, sigma, offset):
    return offset + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def extract_temperature(filename):
    m = re.search(r"Broadened(\d+)C", os.path.basename(filename))
    return int(m.group(1)) if m else None


def fwhm_to_temperature(fwhm_nm, lambda_nm=822.468):
    """
    Doppler FWHM (nm) → kinetic temperature (K).
    FWHM_Doppler = λ * sqrt(8 ln2 * kB*T / (m*c²))
    → T = (FWHM/λ)² * m*c² / (8 ln2 * kB)
    """
    fwhm_m = fwhm_nm * 1e-9
    lam_m  = lambda_nm * 1e-9
    T = (fwhm_m / lam_m) ** 2 * M_CS * C_LIGHT**2 / (8 * np.log(2) * KB)
    return T


# ── data loading ───────────────────────────────────────────────────────────

def load_file(filepath):
    raw  = np.loadtxt(filepath)
    data = {key: raw[:, col] for key, col in COLUMNS.items()}
    data["raw_pmt"] = data["pmt"].copy()
    mask = data["pmt"] == 0
    data["zero_mask"]       = mask
    data["n_zeros_removed"] = int(mask.sum())
    for key in COLUMNS:
        data[key] = data[key][~mask]
    return data


def load_all(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, FILE_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No files matching '{FILE_PATTERN}' in '{data_dir}'")
    dataset = {}
    for f in files:
        temp = extract_temperature(f)
        if temp is None:
            continue
        data = load_file(f)
        data["filepath"] = f
        dataset[temp] = data
        print(f"  Loaded T={temp:3d}°C | {len(data['pmt']):4d} pts "
              f"(removed {data['n_zeros_removed']} zero-count rows)")
    return dataset


# ── analysis ───────────────────────────────────────────────────────────────

def compute_summary(dataset):
    keys = ["temps", "mean_voltage", "std_voltage", "mean_intensity", "std_intensity",
            "mean_power", "peak_pmt", "peak_wavelength"]
    out  = {k: [] for k in keys}
    for temp in sorted(dataset):
        d = dataset[temp]
        mean_v = np.mean(d["voltage"])
        std_v  = np.std(d["voltage"])
        out["temps"].append(temp)
        out["mean_voltage"].append(mean_v)
        out["std_voltage"].append(std_v)
        # Convert voltage to intensity (mW)
        out["mean_intensity"].append(voltage_to_intensity(mean_v))
        # Propagate std: since y = 2.133x - 0.05, std_y = 2.133 * std_x
        out["std_intensity"].append(2.133 * std_v)
        out["mean_power"].append(np.mean(d["power"]))
        out["peak_pmt"].append(np.max(d["pmt"]))
        out["peak_wavelength"].append(d["wavelength"][np.argmax(d["pmt"])])
    return {k: np.array(v) for k, v in out.items()}


def fit_gaussian_peak(wavelength, pmt, smooth_window=11, smooth_poly=3):
    idx = np.argsort(wavelength)
    wl  = wavelength[idx]
    cts = pmt[idx].astype(float)

    cts_s = savgol_filter(cts, smooth_window, smooth_poly) if len(cts) > smooth_window else cts
    peaks, _ = find_peaks(cts_s, height=np.median(cts_s) * 1.3,
                          prominence=np.ptp(cts_s) * 0.05)
    if not len(peaks):
        return None

    best    = peaks[np.argmax(cts_s[peaks])]
    amp0    = float(cts_s[best] - np.min(cts_s))
    center0 = float(wl[best])
    sigma0  = float((wl[-1] - wl[0]) / 8)
    offset0 = float(np.percentile(cts, 10))

    try:
        popt, pcov = curve_fit(
            gaussian, wl, cts,
            p0=[amp0, center0, sigma0, offset0],
            bounds=([0, wl[0], 0, -np.inf], [np.inf, wl[-1], wl[-1]-wl[0], np.inf]),
            maxfev=20000,
        )
        perr  = np.sqrt(np.diag(pcov))
        fwhm  = 2.3548 * abs(popt[2])    # nm
        sigma_err = perr[2]
        fwhm_err  = 2.3548 * sigma_err
        T_kinetic = fwhm_to_temperature(fwhm)
        return {
            "wl_fit":      np.linspace(wl[0], wl[-1], 600),
            "popt":        popt,
            "perr":        perr,
            "amplitude":   popt[0],  "amp_err":    perr[0],
            "center":      popt[1],  "center_err": perr[1],
            "sigma":       abs(popt[2]), "sigma_err": sigma_err,
            "fwhm":        fwhm,     "fwhm_err":   fwhm_err,
            "offset":      popt[3],
            "T_kinetic_K": T_kinetic,
            "T_kinetic_C": T_kinetic - 273.15,
        }
    except RuntimeError:
        return None


# ── plotting ───────────────────────────────────────────────────────────────

def _legend(ax, **kw):
    kw.setdefault("framealpha", 0.8)
    kw.setdefault("labelcolor", TEXT_COLOR)
    leg = ax.legend(**kw)
    if leg:
        plt.setp(leg.get_frame(), facecolor="white", edgecolor="#aaaaaa")
        if leg.get_title():
            leg.get_title().set_color(TEXT_COLOR)
    return leg


def plot_pmt_spectra(dataset, output_dir):
    """Figure 1: Doppler-broadened TPA — PMT fluorescence vs wavelength."""
    temps  = sorted(dataset)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(temps)))
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_COLOR)
    for color, temp in zip(colors, temps):
        d = dataset[temp]; idx = np.argsort(d["wavelength"])
        ax.plot(d["wavelength"][idx], d["pmt"][idx],
                color=color, lw=1.3, alpha=0.85, label=f"{temp}°C")
    set_style(ax)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("PMT Singles (counts)")
    ax.set_title("Doppler-Broadened TPA — PMT Fluorescence vs Wavelength")
    _legend(ax, title="Temperature", fontsize=9, title_fontsize=9, loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "01_pmt_vs_wavelength.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR); plt.close(fig)
    print(f"  Saved: {path}")


def plot_intensity_vs_temperature(summary, output_dir):
    """Figure 2: Mean intensity (mW) vs temperature, converted from photodiode voltage."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.errorbar(summary["temps"], summary["mean_intensity"],
                yerr=summary["std_intensity"], fmt="o-", color=ACCENT_COLOR,
                lw=2, ms=8, markerfacecolor="white", markeredgecolor=ACCENT_COLOR,
                capsize=5, ecolor=ACCENT_COLOR, elinewidth=1.5)
    for t, i in zip(summary["temps"], summary["mean_intensity"]):
        ax.annotate(f"{i:.3f} mW", (t, i), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.5, color="black")
    set_style(ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Mean Intensity (mW)")
    ax.set_title("Mean Intensity vs Temperature")
    ax.set_xticks(summary["temps"])
    fig.tight_layout()
    path = os.path.join(output_dir, "02_intensity_vs_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR); plt.close(fig)
    print(f"  Saved: {path}")


def plot_peak_pmt_vs_temperature(summary, output_dir):
    """Figure 3: Peak PMT fluorescence vs temperature."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.plot(summary["temps"], summary["peak_pmt"], "s-", color="#ff9f7f",
            lw=2, ms=8, markerfacecolor="white", markeredgecolor="#ff9f7f")
    for t, p in zip(summary["temps"], summary["peak_pmt"]):
        ax.annotate(f"{int(p)}", (t, p), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color="black")
    set_style(ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Peak PMT Counts")
    ax.set_title("Peak PMT Fluorescence vs Temperature")
    ax.set_xticks(summary["temps"])
    fig.tight_layout()
    path = os.path.join(output_dir, "03_peak_pmt_vs_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR); plt.close(fig)
    print(f"  Saved: {path}")


def fit_gaussians(dataset):
    """Run Gaussian fits on all temperature datasets and return fit_results dict."""
    fit_results = {}
    for temp in sorted(dataset):
        d   = dataset[temp]
        fit = fit_gaussian_peak(d["wavelength"], d["pmt"])
        fit_results[temp] = fit
    return fit_results


def plot_fwhm_vs_temperature(fit_results, output_dir):
    """
    Figure 4: FWHM vs temperature with theoretical Doppler curve (∝ √T).
    """
    temps_fit = sorted(t for t in fit_results if fit_results[t])
    fwhms     = np.array([fit_results[t]["fwhm"] * 1000 for t in temps_fit])   # pm
    fwhm_errs = np.array([fit_results[t]["fwhm_err"] * 1000 for t in temps_fit])
    temps_K   = np.array(temps_fit) + 273.15

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_COLOR)

    ax.errorbar(temps_fit, fwhms, yerr=fwhm_errs,
                fmt="o", color="#ff9f7f", ms=8, capsize=5,
                ecolor="#ff9f7f", markerfacecolor="white",
                markeredgecolor="#ff9f7f", zorder=5, label="Measured FWHM")

    # Fit FWHM = A * √T
    def sqrt_model(T_K, A):
        return A * np.sqrt(T_K)

    try:
        popt, pcov = curve_fit(sqrt_model, temps_K, fwhms / 1000,
                               p0=[1e-5], sigma=fwhm_errs / 1000)
        T_fine = np.linspace(temps_K.min() - 10, temps_K.max() + 10, 300)
        fwhm_theory = sqrt_model(T_fine, *popt) * 1000
        ax.plot(T_fine - 273.15, fwhm_theory, "--", color=ACCENT_COLOR,
                lw=1.8, alpha=0.9, label=r"$A\sqrt{T}$ fit")
    except Exception:
        pass

    set_style(ax)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("FWHM (pm)")
    ax.set_title(r"Doppler FWHM vs Temperature  ($\propto \sqrt{T}$ expected)")
    ax.set_xticks(temps_fit)
    _legend(ax, fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "04_fwhm_vs_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR); plt.close(fig)
    print(f"  Saved: {path}")


# ── report ─────────────────────────────────────────────────────────────────

def print_report(dataset, summary, fit_results):
    sep = "─" * 95
    print(f"\n{sep}")
    print("  DOPPLER-BROADENED TPA — ANALYSIS REPORT")
    print(sep)
    print(f"\n{'T':>5} │ {'N':>5} │ {'Zeros':>5} │ {'<I>(mW)':>9} │ "
          f"{'σI(mW)':>8} │ {'PkPMT':>7} │ {'FWHM(pm)':>10} │ {'T_kin(°C)':>10}")
    print("─" * 85)
    for i, temp in enumerate(sorted(dataset)):
        d   = dataset[temp]
        fit = fit_results.get(temp)
        fwhm_str = f"{fit['fwhm']*1000:8.2f} ± {fit['fwhm_err']*1000:.2f}" if fit else "   fit failed"
        tkin_str = f"{fit['T_kinetic_C']:10.1f}" if fit else "          —"
        print(f"  {temp:3d}°C │ {len(d['pmt']):5d} │ {d['n_zeros_removed']:5d} │ "
              f"{summary['mean_intensity'][i]:9.4f} │ "
              f"{summary['std_intensity'][i]:8.4f} │ "
              f"{summary['peak_pmt'][i]:7.0f} │ {fwhm_str} │ {tkin_str}")
    print(sep + "\n")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # Default data_dir is the 'data' subfolder relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "data")

    parser = argparse.ArgumentParser()
    default_images_dir = os.path.join(script_dir, "images")

    parser.add_argument("--data_dir",   default=default_data_dir)
    parser.add_argument("--output_dir", default=default_images_dir)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading data from: {args.data_dir}")
    dataset = load_all(args.data_dir)

    print("\nComputing summary statistics...")
    summary = compute_summary(dataset)

    print("\nFitting Gaussian peaks...")
    fit_results = fit_gaussians(dataset)

    print("\nGenerating plots...")
    plot_pmt_spectra(dataset, args.output_dir)
    plot_intensity_vs_temperature(summary, args.output_dir)
    plot_peak_pmt_vs_temperature(summary, args.output_dir)
    plot_fwhm_vs_temperature(fit_results, args.output_dir)

    print_report(dataset, summary, fit_results)
    print(f"All outputs saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
