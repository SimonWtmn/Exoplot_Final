import numpy as np
import batman
import emcee
import multiprocessing
import os
from tqdm import tqdm


from utils.helpers import find_bounds

# ===========================================================
# Transit Model
# ===========================================================

def make_transit_model(time, theta):
    t0, per, rp, a, inc, u1, u2 = theta[:7]
    coeff = [u1, u2]

    transit_params = batman.TransitParams()
    transit_params.t0 = t0
    transit_params.per = per
    transit_params.rp = rp
    transit_params.a = a
    transit_params.inc = inc
    transit_params.ecc = 0
    transit_params.w = 90
    transit_params.u = coeff
    transit_params.limb_dark = "quadratic"

    model = batman.TransitModel(transit_params, time)
    return model.light_curve(transit_params)


# ===========================================================
# Bayes Analysis 
# ===========================================================

def log_prior(params, BOUNDS):
    """ 
    Compute the prior probability of the model.
    """
    for p, (low, high) in zip(params, BOUNDS):
        if not (low <= p <= high):
            return -np.inf
    return 0.0


def log_likelihood(params, time, flux, flux_err):
    """
    Compute log-likelihood of transit model given observed data.
    """
    try:
        try:
            model_flux = make_transit_model(time, params)
        except Exception:
            return -np.inf

        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2
    except Exception as e:
        print(f"[log_likelihood ERROR] params={params} -> {e}")
        return -1e20


def log_posterior(params, time, flux, flux_err, BOUNDS):
    try:
        lp = log_prior(params, BOUNDS)
        if not np.isfinite(lp):
            return -1e20
        ll = log_likelihood(params, time, flux, flux_err)
        if not np.isfinite(ll):
            return -1e20
        return lp + ll
    except Exception as e:
        print(f"[log_posterior ERROR] params={params} -> {e}")
        return -np.inf


def neg_log_posterior(params, time, flux, flux_err, BOUNDS):
    return -log_posterior(params, time, flux, flux_err, BOUNDS)


# ===========================================================
# Run MCMC
# ===========================================================

def mcmc(p0, nwalkers, steps, ndim, log_posterior, data):
    """
    Run MCMC with burn-in and production phases.
    """
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=data)

    # ---------------------------
    # Burn-in phase
    # ---------------------------
    print("Running burn-in...")
    burnin_steps = steps // 10

    for _ in tqdm(sampler.sample(p0, iterations=burnin_steps), total=burnin_steps, desc="Burn-in"):
        pass

    p_burnin = sampler.get_last_sample().coords

    mean_acceptance_burnin = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction (burn-in): {mean_acceptance_burnin:.3f} (recommended ~0.2–0.5)")

    sampler.reset()

    # ---------------------------
    # Production phase
    # ---------------------------
    print("Running production...")
    for _ in tqdm(sampler.sample(p_burnin, iterations=steps), total=steps, desc="Production"):
        pass

    mean_acceptance_prod = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction (production): {mean_acceptance_prod:.3f} (recommended ~0.2–0.5)")

    try:
        tau = sampler.get_autocorr_time()
        print(f"Estimated autocorrelation time per parameter: {tau}")
        burnin_est = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
    except emcee.autocorr.AutocorrError:
        print("Warning: chain too short to reliably estimate autocorrelation time.")
        burnin_est = 100
        thin = 10

    pos = sampler.get_last_sample().coords
    prob = sampler.get_last_sample().log_prob
    state = sampler.get_last_sample().random_state

    return sampler, pos, prob, state