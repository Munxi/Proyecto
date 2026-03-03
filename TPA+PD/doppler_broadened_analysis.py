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
import matplotlib.gridspec as gridspec
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
DARK_BG      = "#0f0f1a"
PANEL_BG     = "#1a1a2e"
TEXT_COLOR   = "white"
ACCENT_COLOR = "#7ecfff"

FILE_PATTERN = "DopplerBroadened*C*.txt"

COLUMNS = {"power": 0, "wavelength": 1, "pmt": 2, "frequency": 3, "voltage": 4}


# ── helpers ────────────────────────────────────────────────────────────────

def set_dark_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.spines[:].set_color("#444466")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.2, color="gray")
    ax.grid(True, which="minor", alpha=0.07, color="gray")
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
    keys = ["temps","mean_voltage","std_voltage","mean_power","peak_pmt","peak_wavelength"]
    out  = {k: [] for k in keys}
    for temp in sorted(dataset):
        d = dataset[temp]
        out["temps"].append(temp)
        out["mean_voltage"].append(np.mean(d["voltage"]))
        out["std_voltage"].append(np.std(d["voltage"]))
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
    kw.setdefault("framealpha", 0.35)
    kw.setdefault("labelcolor", TEXT_COLOR)
    leg = ax.legend(**kw)
    if leg:
        plt.setp(leg.get_frame(), facecolor="#2a2a4a", edgecolor="#555577")
        if leg.get_title():
            leg.get_title().set_color(TEXT_COLOR)
    return leg


def plot_pmt_spectra(dataset, output_dir):
    temps  = sorted(dataset)
    colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(temps)))
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)
    for color, temp in zip(colors, temps):
        d = dataset[temp]; idx = np.argsort(d["wavelength"])
        ax.plot(d["wavelength"][idx], d["pmt"][idx],
                color=color, lw=1.3, alpha=0.85, label=f"{temp}°C")
    set_dark_style(ax)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("PMT Singles (counts)")
    ax.set_title("Doppler-Broadened TPA — PMT Fluorescence vs Wavelength")
    _legend(ax, title="Temperature", fontsize=9, title_fontsize=9, loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "01_pmt_vs_wavelength.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


def plot_voltage_vs_temperature(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.errorbar(summary["temps"], summary["mean_voltage"],
                yerr=summary["std_voltage"], fmt="o-", color=ACCENT_COLOR,
                lw=2, ms=8, markerfacecolor="white", markeredgecolor=ACCENT_COLOR,
                capsize=5, ecolor=ACCENT_COLOR, elinewidth=1.5)
    for t, v in zip(summary["temps"], summary["mean_voltage"]):
        ax.annotate(f"{v:.3f} V", (t, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.5, color="#aaddff")
    set_dark_style(ax)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Mean Photodiode Voltage (V)")
    ax.set_title("Mean Photodiode Voltage vs Temperature")
    ax.set_xticks(summary["temps"])
    fig.tight_layout()
    path = os.path.join(output_dir, "02_voltage_vs_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


def plot_peak_pmt_vs_temperature(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(summary["temps"], summary["peak_pmt"], "s-", color="#ff9f7f",
            lw=2, ms=8, markerfacecolor="white", markeredgecolor="#ff9f7f")
    for t, p in zip(summary["temps"], summary["peak_pmt"]):
        ax.annotate(f"{int(p)}", (t, p), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color="#ffccaa")
    set_dark_style(ax)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Peak PMT Counts")
    ax.set_title("Peak PMT Fluorescence vs Temperature")
    ax.set_xticks(summary["temps"])
    fig.tight_layout()
    path = os.path.join(output_dir, "03_peak_pmt_vs_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


def plot_gaussian_fits(dataset, output_dir):
    temps  = sorted(dataset)
    ncols  = 2; nrows = (len(temps) + 1) // ncols
    colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(temps)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    fit_results = {}
    for i, (temp, color) in enumerate(zip(temps, colors)):
        ax = axes[i // ncols][i % ncols]
        d  = dataset[temp]; idx = np.argsort(d["wavelength"])
        wl = d["wavelength"][idx]; cts = d["pmt"][idx]
        ax.plot(wl, cts, ".", color=color, ms=4, alpha=0.7, label="Data")
        fit = fit_gaussian_peak(d["wavelength"], d["pmt"])
        fit_results[temp] = fit
        if fit:
            wlf = fit["wl_fit"]
            ax.plot(wlf, gaussian(wlf, *fit["popt"]),
                    color="white", lw=1.8, alpha=0.9, label="Gaussian fit")
            info = (f"λ₀ = {fit['center']:.6f} nm\n"
                    f"FWHM = {fit['fwhm']*1000:.2f} ± {fit['fwhm_err']*1000:.2f} pm\n"
                    f"T_kin = {fit['T_kinetic_C']:.1f}°C")
            ax.text(0.03, 0.97, info, transform=ax.transAxes,
                    color="white", fontsize=8.5, va="top",
                    bbox=dict(facecolor="#2a2a4a", alpha=0.75, edgecolor="none"))
        set_dark_style(ax)
        ax.set_title(f"T = {temp}°C", color=color, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("PMT Counts")
        _legend(ax, fontsize=8)

    for j in range(len(temps), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Gaussian Fits to Doppler-Broadened TPA Peaks",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "04_gaussian_fits.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")
    return fit_results


def plot_fwhm_vs_temperature(fit_results, output_dir):
    """
    FWHM vs temperature with theoretical Doppler curve:
    FWHM(T) ∝ √T  →  fit A*√T to data.
    """
    temps_fit = sorted(t for t in fit_results if fit_results[t])
    fwhms     = np.array([fit_results[t]["fwhm"] * 1000 for t in temps_fit])   # pm
    fwhm_errs = np.array([fit_results[t]["fwhm_err"] * 1000 for t in temps_fit])
    temps_K   = np.array(temps_fit) + 273.15

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)

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

    set_dark_style(ax)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("FWHM (pm)")
    ax.set_title(r"Doppler FWHM vs Temperature  ($\propto \sqrt{T}$ expected)")
    ax.set_xticks(temps_fit)
    _legend(ax, fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "05_fwhm_vs_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


def plot_kinetic_vs_setpoint(fit_results, output_dir):
    """Kinetic temperature from FWHM vs thermocouple set-point."""
    temps_fit = sorted(t for t in fit_results if fit_results[t])
    T_kin_C   = [fit_results[t]["T_kinetic_C"] for t in temps_fit]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)

    # 1:1 reference line
    lo, hi = min(temps_fit) - 5, max(temps_fit) + 5
    ax.plot([lo, hi], [lo, hi], "--", color="#555577", lw=1.5, label="T_kin = T_set")

    ax.scatter(temps_fit, T_kin_C, color="#aaffaa", s=80, zorder=5,
               edgecolors="white", linewidths=0.8, label="T from Doppler FWHM")
    for ts, tk in zip(temps_fit, T_kin_C):
        ax.annotate(f"{tk:.1f}°C", (ts, tk), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color="#ccffcc")

    set_dark_style(ax)
    ax.set_xlabel("Set-point Temperature (°C)")
    ax.set_ylabel("Kinetic Temperature from FWHM (°C)")
    ax.set_title("Kinetic Temperature (Doppler) vs Thermocouple Set-point")
    _legend(ax, fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "06_kinetic_vs_setpoint.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


def plot_power_vs_wavelength(dataset, output_dir):
    temps  = sorted(dataset)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(temps)))
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    for color, temp in zip(colors, temps):
        d = dataset[temp]; idx = np.argsort(d["wavelength"])
        ax.plot(d["wavelength"][idx], d["power"][idx],
                color=color, lw=1.1, alpha=0.8, label=f"{temp}°C")
    set_dark_style(ax)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Power (mW)")
    ax.set_title("Laser Power vs Wavelength (Scan Uniformity Check)")
    _legend(ax, title="Temperature", fontsize=9, title_fontsize=9, loc="best")
    fig.tight_layout()
    path = os.path.join(output_dir, "07_power_vs_wavelength.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


def plot_dashboard(dataset, summary, fit_results, output_dir):
    temps  = sorted(dataset)
    colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(temps)))

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A — PMT spectra
    ax_a = fig.add_subplot(gs[0, :2])
    for color, temp in zip(colors, temps):
        d = dataset[temp]; idx = np.argsort(d["wavelength"])
        ax_a.plot(d["wavelength"][idx], d["pmt"][idx],
                  color=color, lw=1.2, alpha=0.8, label=f"{temp}°C")
    set_dark_style(ax_a)
    ax_a.set_xlabel("Wavelength (nm)"); ax_a.set_ylabel("PMT Counts")
    ax_a.set_title("A — Doppler-Broadened PMT Spectra")
    _legend(ax_a, title="T", fontsize=7.5, title_fontsize=8,
            framealpha=0.3, loc="upper right", ncol=2)

    # B — voltage vs T
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.errorbar(summary["temps"], summary["mean_voltage"],
                  yerr=summary["std_voltage"], fmt="o-", color=ACCENT_COLOR,
                  lw=2, ms=6, markerfacecolor="white", markeredgecolor=ACCENT_COLOR,
                  capsize=4, ecolor=ACCENT_COLOR)
    set_dark_style(ax_b)
    ax_b.set_xlabel("Temperature (°C)"); ax_b.set_ylabel("Mean PD Voltage (V)")
    ax_b.set_title("B — Photodiode Voltage vs T")
    ax_b.set_xticks(summary["temps"]); plt.setp(ax_b.get_xticklabels(), rotation=45, fontsize=7)

    # C — FWHM vs T
    ax_c = fig.add_subplot(gs[1, 0])
    temps_fit = sorted(t for t in fit_results if fit_results[t])
    fwhms     = np.array([fit_results[t]["fwhm"] * 1000 for t in temps_fit])
    fwhm_errs = np.array([fit_results[t]["fwhm_err"] * 1000 for t in temps_fit])
    ax_c.errorbar(temps_fit, fwhms, yerr=fwhm_errs, fmt="s-", color="#ff9f7f",
                  lw=2, ms=7, markerfacecolor="white", markeredgecolor="#ff9f7f",
                  capsize=4, ecolor="#ff9f7f")
    try:
        def sqrt_model(T_K, A): return A * np.sqrt(T_K)
        popt, _ = curve_fit(sqrt_model, np.array(temps_fit) + 273.15, fwhms / 1000,
                            p0=[1e-5], sigma=fwhm_errs / 1000)
        T_fine = np.linspace(min(temps_fit) + 273.15 - 10, max(temps_fit) + 273.15 + 10, 200)
        ax_c.plot(T_fine - 273.15, sqrt_model(T_fine, *popt) * 1000, "--",
                  color=ACCENT_COLOR, lw=1.5, alpha=0.8, label=r"$A\sqrt{T}$")
        _legend(ax_c, fontsize=8)
    except Exception:
        pass
    set_dark_style(ax_c)
    ax_c.set_xlabel("Temperature (°C)"); ax_c.set_ylabel("FWHM (pm)")
    ax_c.set_title(r"C — Doppler FWHM vs T")
    ax_c.set_xticks(temps_fit); plt.setp(ax_c.get_xticklabels(), rotation=45, fontsize=7)

    # D — peak PMT vs T
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(summary["temps"], summary["peak_pmt"], "D-", color="#b0ff80",
              lw=2, ms=7, markerfacecolor="white", markeredgecolor="#b0ff80")
    set_dark_style(ax_d)
    ax_d.set_xlabel("Temperature (°C)"); ax_d.set_ylabel("Peak PMT Counts")
    ax_d.set_title("D — Peak Fluorescence vs T")
    ax_d.set_xticks(summary["temps"]); plt.setp(ax_d.get_xticklabels(), rotation=45, fontsize=7)

    # E — kinetic vs set-point
    ax_e = fig.add_subplot(gs[1, 2])
    T_kin = [fit_results[t]["T_kinetic_C"] for t in temps_fit]
    lo, hi = min(temps_fit) - 5, max(temps_fit) + 5
    ax_e.plot([lo, hi], [lo, hi], "--", color="#555577", lw=1.3, label="1:1")
    ax_e.scatter(temps_fit, T_kin, color="#aaffaa", s=60, zorder=5,
                 edgecolors="white", linewidths=0.7)
    set_dark_style(ax_e)
    ax_e.set_xlabel("Set-point T (°C)"); ax_e.set_ylabel("Kinetic T (°C)")
    ax_e.set_title("E — Kinetic T vs Set-point T")
    _legend(ax_e, fontsize=8)

    fig.suptitle("Cs Doppler-Broadened TPA Spectroscopy @ 822 nm — Summary",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.99)
    path = os.path.join(output_dir, "00_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close(fig)
    print(f"  Saved: {path}")


# ── report ─────────────────────────────────────────────────────────────────

def print_report(dataset, summary, fit_results):
    sep = "─" * 85
    print(f"\n{sep}")
    print("  DOPPLER-BROADENED TPA — ANALYSIS REPORT")
    print(sep)
    print(f"\n{'T':>5} │ {'N':>5} │ {'Zeros':>5} │ {'<V>(V)':>8} │ "
          f"{'σV(V)':>7} │ {'PkPMT':>7} │ {'FWHM(pm)':>10} │ {'T_kin(°C)':>10}")
    print("─" * 75)
    for i, temp in enumerate(sorted(dataset)):
        d   = dataset[temp]
        fit = fit_results.get(temp)
        fwhm_str = f"{fit['fwhm']*1000:8.2f} ± {fit['fwhm_err']*1000:.2f}" if fit else "   fit failed"
        tkin_str = f"{fit['T_kinetic_C']:10.1f}" if fit else "          —"
        print(f"  {temp:3d}°C │ {len(d['pmt']):5d} │ {d['n_zeros_removed']:5d} │ "
              f"{summary['mean_voltage'][i]:8.4f} │ "
              f"{summary['std_voltage'][i]:7.4f} │ "
              f"{summary['peak_pmt'][i]:7.0f} │ {fwhm_str} │ {tkin_str}")
    print(sep + "\n")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default=".")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading data from: {args.data_dir}")
    dataset = load_all(args.data_dir)

    print("\nComputing summary statistics...")
    summary = compute_summary(dataset)

    print("\nGenerating plots...")
    plot_pmt_spectra(dataset, args.output_dir)
    plot_voltage_vs_temperature(summary, args.output_dir)
    plot_peak_pmt_vs_temperature(summary, args.output_dir)
    fit_results = plot_gaussian_fits(dataset, args.output_dir)
    plot_fwhm_vs_temperature(fit_results, args.output_dir)
    plot_kinetic_vs_setpoint(fit_results, args.output_dir)
    plot_power_vs_wavelength(dataset, args.output_dir)
    plot_dashboard(dataset, summary, fit_results, args.output_dir)

    print_report(dataset, summary, fit_results)
    print(f"All outputs saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
