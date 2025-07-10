from scipy.optimize import minimize
from utils.plotting import *
from utils.helpers import find_bounds, compute_bounds
from utils.mcmc import *

from matplotlib.backends.backend_pdf import PdfPages
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ===========================================================
    # Load Light Curve
    # ===========================================================

    target_name = "WASP 76"
    search_result = lk.search_lightcurve(target=target_name)
    row = 0
    lc = search_result[row].download().normalize().remove_nans()

    time     = lc.time.value
    flux     = lc.flux.value
    flux_err = lc.flux_err.value

    # ===========================================================
    # Compute Periodogram
    # ===========================================================

    pg = lc.to_periodogram(method='bls', minimum_period=0.4)

    freq      = pg.frequency.value
    per       = pg.period.value 
    power     = pg.power.value
    max_freq  = pg.frequency_at_max_power.value
    max_per   = pg.period_at_max_power.value

    # ===========================================================
    # Compute Bounds and Initial Guess
    # ===========================================================

    # pl_name = "WASP-76 b"
    # BOUNDS_perfect = find_bounds(pl_name, 10)

    BOUNDS = compute_bounds(time, flux, flux_err, max_per)

    # print("Perfet bounds :", BOUNDS_perfect)
    print("Computed bounds :", BOUNDS)

    nwalkers = 46
    steps = 500

    x0 = np.array([time[np.argmin(flux)], max_per, np.sqrt(1 - np.min(flux)) - np.sqrt(np.mean(flux_err)), 2.8*max_per**(2/3), 89.0, 0.5, 0.5])
    # x0_perfect = np.array([(b[0] + b[1]) / 2 for b in BOUNDS_perfect])

    # print("Perfect Initial :", x0_perfect)
    print("Computed Initial :", x0)
    # ===========================================================
    # Find Maximum a Posteriori Estimate
    # ===========================================================

    res = minimize(neg_log_posterior, x0, args=(time, flux, flux_err, BOUNDS), method='L-BFGS-B')
    theta_map = res.x

    # res_perfect = minimize(neg_log_posterior, x0, args=(time, flux, flux_err, BOUNDS_perfect), method='POWELL')
    # theta_map_perfect = res_perfect.x

    # print("Perfect Optimized Initial :", theta_map_perfect)
    print("Computed Optimized Initial :", theta_map)

    # ===========================================================
    # Prepare MCMC Sampler
    # ===========================================================

    ndim = len(theta_map)

    spread = 0.005
    p0 = []
    for _ in range(nwalkers):
        walker = theta_map + spread * np.random.randn(ndim)
        walker = np.clip(walker, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
        p0.append(walker)
    p0 = np.array(p0)

    data = (time, flux, flux_err, BOUNDS)

    # ===========================================================
    # Run MCMC
    # ===========================================================

    sampler, pos, prob, state = mcmc(p0, nwalkers, steps, ndim, log_posterior, data)

    # ===========================================================
    # Analyze Chain and Extract Best Parameters
    # ===========================================================

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()

    samples = chain.reshape(-1, chain.shape[-1])
    flat_log_prob = log_prob.reshape(-1)

    theta_max = samples[np.argmax(flat_log_prob)]

    # ===========================================================
    # Fold Light Curve at Best-fit Epoch
    # ===========================================================

    fold = lc.fold(period=max_per, epoch_time=theta_max[0])

    time_fold     = fold.time.value
    flux_fold     = fold.flux.value
    flux_fold_err = fold.flux_err.value


    # ===========================================================
    # Plots
    # ===========================================================

    figures = []

    # Raw lightcurve
    fig = lightcurve(time, flux, flux_err, target_name, 'Raw')
    figures.append(('Raw Lightcurve', fig_to_base64(fig)))

    # Periodogram
    fig = periodogram(freq, per, power, max_freq, max_per, target_name)
    figures.append(('Periodogram', fig_to_base64(fig)))

    # Model fit
    fig = model(time, flux, flux_err, time_fold, flux_fold, flux_fold_err, target_name, samples, flat_log_prob)
    figures.append(('Model Fit', fig_to_base64(fig)))

    # Evolution
    fig = evolution(chain)
    figures.append(('Walkers Evolution', fig_to_base64(fig)))

    # Corner
    fig = corner_plot(samples)
    figures.append(('Corner Plot', fig_to_base64(fig)))

    # ===========================================================
    # Generate HTML Report with Figures
    # ===========================================================

    with PdfPages("report.pdf") as pdf:
        plt.rcParams.update({'font.size': 12})
        for title, fig in [
            ('Raw Lightcurve', lightcurve(time, flux, flux_err, target_name, 'Raw')),
            ('Periodogram', periodogram(freq, per, power, max_freq, max_per, target_name)),
            ('Model Fit', model(time, flux, flux_err, time_fold, flux_fold, flux_fold_err, target_name, samples, flat_log_prob)),
            ('Walkers Evolution', evolution(chain)),
            ('Corner Plot', corner_plot(samples)),
        ]:
            fig.suptitle(title)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print("Analysis complete. See report.pdf for results.")



if __name__ == "__main__":
    main()
