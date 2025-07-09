import matplotlib.pyplot as plt
import numpy as np
import corner
import io
import base64

from utils.mcmc import make_transit_model
from utils.helpers import LABELS


# ===========================================================
# Plot Light Curves
# ===========================================================

def lightcurve(time, flux, flux_err, target_name, kind):
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(15, 5))

    axs[0].plot(time, flux, color="navy", alpha=0.8)
    axs[0].set_title("Light Curve (Line)", fontsize=12)

    axs[1].errorbar(time, flux, yerr=flux_err, linestyle="None", marker=".", ms=2, color="red", ecolor="orange", elinewidth=0.5, alpha=0.7, capsize=1.5)
    axs[1].set_title("Light Curve (With Errors)", fontsize=12)

    for ax in axs:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=10)

    fig.suptitle(f"{kind} Light of {target_name}", fontsize=14)
    fig.supxlabel("Time (JD)", fontsize=12)
    fig.supylabel("Relative Flux ($e^{-}s^{-1}$)", fontsize=12)

    return fig



# ===========================================================
# Plot Periodogram
# ===========================================================

def periodogram(freq, per, power, max_freq, max_per, target_name):
    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

    axs[0].plot(freq, power, color="navy", alpha=0.8)
    axs[0].axvline(max_freq, color="orange", linestyle="--", label=f"Frequency at max power: {max_freq:.4f} 1/d")
    axs[0].set_xlabel("Frequency $\\left( \\frac{1}{d} \\right)$", fontsize=12)
    axs[0].set_title("Periodogram (Frequency)", fontsize=12)

    axs[1].errorbar(per, power, color="navy", alpha=0.8)
    axs[1].axvline(max_per, color="orange", linestyle="--", label=f"Period at max power: {max_per:.4f} d")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Period ($d$)", fontsize=12)
    axs[1].set_title("Periodogram (Period)", fontsize=12)

    for ax in axs:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(f"Periodogram of {target_name}", fontsize=14)
    fig.supylabel("BlS Power", fontsize=12)

    return fig



# ===========================================================
# Plot Fit Comparaison with model
# ===========================================================

def model(time, flux, flux_err, time_fold, flux_fold, flux_fold_err, target_name, samples, flat_log_prob):
    theta_max = samples[np.argmax(flat_log_prob)]

    phase_min, phase_max = theta_max[0]-0.2, theta_max[0]+0.2

    mask = (time_fold >= phase_min) & (time_fold <= phase_max)

    time_fold = time_fold[mask]
    flux_fold = flux_fold[mask]
    flux_fold_err = flux_fold_err[mask]

    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(18, 10))

    axs[0][0].errorbar(time, flux, yerr=flux_err, linestyle="None", marker=".", ms=2, color="red", ecolor="orange", elinewidth=0.5, alpha=0.7, capsize=1.5)
    axs[0][0].plot(time, make_transit_model(time, theta_max), color="navy", zorder=3)
    axs[0][0].set_title("Fit on Raw Light Curve", fontsize=12)

    axs[0][1].errorbar(time_fold, flux_fold, yerr=flux_fold_err, linestyle="None", marker=".", ms=2, color="red", ecolor="orange", elinewidth=0.5, alpha=0.7, capsize=1.5)
    axs[0][1].plot(time_fold, make_transit_model(time_fold, theta_max), color="navy", zorder=3)
    axs[0][1].set_title("Fit on Folded Light Curve", fontsize=12)

    axs[1][0].errorbar(time_fold, flux_fold, yerr=flux_fold_err, linestyle="None", marker=".", ms=2, color="red", ecolor="orange", elinewidth=0.5, alpha=0.7, capsize=1.5)
    for theta in samples[np.random.randint(len(samples), size=100)]:
            axs[1][0].plot(time_fold, make_transit_model(time_fold, theta), color="navy", alpha=0.1, zorder=3)
    axs[1][0].set_title("Walkers' Tryouts", fontsize=12) 

    param_text = ""
    for i in range(samples.shape[1]):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        param_text += f"{LABELS[i]} = {mcmc[1]:.5f} (+{q[1]:.5f}/-{q[0]:.5f})\n"

    axs[1,1].axis("off")
    axs[1,1].text(0.05, 0.95, param_text, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                transform=axs[1,1].transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


    for ax in axs[0::2].flatten():
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.set_xlabel("Time (JD)", fontsize=10)
        ax.set_ylabel("Relative Flux ($e^{-}s^{-1}$)", fontsize=10)

    fig.suptitle(f"Results of MCMC for {target_name}", fontsize=14)

    return fig



# ===========================================================
# Plot Walkers'Evolution 
# ===========================================================

def evolution(chain):
    nsteps, nwalkers, ndim = chain.shape

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    for i in range(ndim):
        ax = axes[i]
        for j in range(nwalkers):
            ax.plot(chain[:, j, i], alpha=0.5, lw=0.5)
        ax.set_ylabel(f"{LABELS[i]}")
        ax.grid()

    axes[-1].set_xlabel("Step")

    return fig



# ===========================================================
# Plot Corner
# ===========================================================

def corner_plot(samples):
    fig = corner.corner(samples, labels=LABELS, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    return fig



# ===========================================================
# Translate fig to HTML
# ===========================================================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"