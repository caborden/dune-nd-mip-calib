import numpy as np
import os
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable: iterable
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import collections
from functools import partial
import pickle
import h5py

def landau(x, mpv, eta):
    # Upstream Moyal-like approximation to Landau (normalization absorbed by A).
    # mpv: Most Probable Value
    # eta: scale parameter (related to width)
    xi = (x - mpv) / eta
    return np.exp(-0.5 * (xi + np.exp(-xi))) / eta

def gaussian(x, mean, sigma, A):
    return A*np.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def langau(x, mpv, eta, sigma, A):
    # x: points in charge, adc, energy
    # mpv: Landau MPV
    # eta: Landau width
    # sigma: Gaussian sigma
    # A: amplitude
    step = 0.01  # integration resolution
    conv = np.zeros_like(x)
    xrange = np.arange(-20*sigma, 20*sigma, step)

    for i, xi in enumerate(x):
        y = landau(xi - xrange, mpv, eta) * gaussian(xrange, 0, sigma, 1)
        conv[i] = trapezoid(y, dx=step)

    return A * conv

def compute_fwhm(x_vals, y_vals):
    half_max = np.max(y_vals) / 2.0
    above = y_vals > half_max
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    if len(crossings) >= 2:
        x1 = np.interp(
            half_max,
            [y_vals[crossings[0]], y_vals[crossings[0] + 1]],
            [x_vals[crossings[0]], x_vals[crossings[0] + 1]]
        )
        x2 = np.interp(
            half_max,
            [y_vals[crossings[-1] + 1], y_vals[crossings[-1]]],
            [x_vals[crossings[-1] + 1], x_vals[crossings[-1]]]
        )
        return x2 - x1
    return np.nan

def fit_histogram(
    data,
    fit_func,
    p0,
    fit_range,
    bin_range,
    nbins=100,
    histogram=None,
    max_width=None
):
    # Prebinned input allows bounded-memory analysis of merged files.
    counts, edges = (np.histogram(data, bins=nbins, range=bin_range)
                     if histogram is None else histogram)
    x_hist = 0.5 * (edges[:-1] + edges[1:])
    fit_mask = (x_hist > fit_range[0]) & (x_hist < fit_range[1])

    fit_x = x_hist[fit_mask]
    fit_counts = counts[fit_mask]

    sigma_bins = np.sqrt(fit_counts)
    sigma_bins[sigma_bins == 0] = 1.0

    if p0 is None:
        guess_mpv = fit_x[np.argmax(fit_counts)]
        spread = (np.std(data) if histogram is None else
                  np.sqrt(np.average((x_hist-np.average(x_hist,weights=counts))**2,weights=counts)))
        guess_eta = max(spread / 2.0, 0.1)
        guess_sigma = max(spread / 4.0, 0.1)
        guess_A = np.max(fit_counts) * guess_eta
        p0 = [guess_mpv, guess_eta, guess_sigma, guess_A]

    # order [mpv/mean, eta, sigma, amplitude]
    lower_bounds = [fit_range[0], 1e-5, 1e-5, 0]
    width_bound = np.inf if max_width is None else max_width
    upper_bounds = [fit_range[1], width_bound, width_bound, np.inf]

    try:
        params, cov = curve_fit(
            fit_func,
            fit_x,
            fit_counts,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            sigma=sigma_bins,
            absolute_sigma=True,
            maxfev=10000
        )

        fit_vals = fit_func(fit_x, *params)

        chi2 = np.sum(((fit_counts - fit_vals) / sigma_bins) ** 2)
        dof = len(fit_x) - len(params)
        chi2_red = chi2 / dof

        mpv_err = np.sqrt(cov[0, 0]) if cov is not None else np.nan

        return {
            "params": params,
            "cov": cov,
            "mpv_error": mpv_err,
            "chi2_red": chi2_red,
            "x_hist": x_hist,
            "x_fit": fit_x,
            "counts": counts,
            "fit_vals": fit_vals
        }

    except (RuntimeError, ValueError, FloatingPointError) as e:
        print(f"Fit failed: {e}")
        return None

def analyze_dqdx_by_bin(
    x,
    dq,
    dx,
    x_bin_edges,
    fit_range=(20,80),
    dqdx_range=(0,120),
    nbins=100,
    min_entries=10
):

    dqdx = dq / dx

    x_bin_idx = np.digitize(x, x_bin_edges) - 1
    n_bins = len(x_bin_edges) - 1
    bin_centers = 0.5 * (x_bin_edges[:-1] + x_bin_edges[1:])

    mpvs = np.full(n_bins, np.nan)
    bins = np.full(n_bins, np.nan)
    sigmas = np.full(n_bins, np.nan)
    chi2s = np.full(n_bins, np.nan)
    fwhms = np.full(n_bins, np.nan)
    mpv_errors = np.full(n_bins, np.nan)

    for ibin in tqdm(range(n_bins)):

        mask = x_bin_idx == ibin
        dqdx_vals = dqdx[mask]

        if len(dqdx_vals) < min_entries:
            continue

        plt.figure()
        plt.hist(dqdx_vals, bins=nbins, range=dqdx_range, alpha=0.6, label="Data")
        plt.xlabel("dQ/dX [ke-/cm]")
        plt.ylabel("Counts")
        plt.savefig(f"fit_xbin_{bin_centers[ibin]:.2f}.pdf")

        result = fit_histogram(
            dqdx_vals,
            langau,
            p0=None,
            fit_range=fit_range,
            bin_range=dqdx_range,
            nbins=nbins
        )

        if result is None:
            continue

        params = result["params"]
        fit_vals = result["fit_vals"]
        x_hist = result["x_hist"]
        x_fit = result["x_fit"]

        mpvs[ibin] = params[0]
        mpv_errors[ibin] = result["mpv_error"]
        sigmas[ibin] = params[2]
        chi2s[ibin] = result["chi2_red"]
        fwhms[ibin] = compute_fwhm(x_fit, fit_vals)
        bins[ibin] = bin_centers[ibin]

        plt.plot(x_fit, fit_vals, 'r-', label="Fit")

        plt.title(
            f"Bin Center: {bin_centers[ibin]:.2f}\n"
            f"MPV={params[0]:.2f} "
            f"FWHM={fwhms[ibin]:.2f} "
            f"Chi2/ndf={result['chi2_red']:.2f}"
        )

        plt.savefig(f"fit_xbin_{bin_centers[ibin]:.2f}.pdf")
        plt.close()


    return {
            "mpv": mpvs,
            "mpv_error": mpv_errors,
            "chi2_red": chi2s,
            "bin": bins,
            "sigma": sigmas,
            "fwhm": fwhms
        }

def plot_dqdx_map_theta_phi(
    theta,
    phi,
    dq,
    dx,
    theta_bin_edges,
    phi_bin_edges,
    statistic='mean',
    cmap='viridis',
    vmin=None,
    vmax=None,
    maxnorm=False,
    title='dQ/dx vs θ and φ',
    min_entries=10
):
    import numpy as np
    import matplotlib.pyplot as plt

    dqdx = dq / dx

    n_theta = len(theta_bin_edges) - 1
    n_phi = len(phi_bin_edges) - 1

    dqdx_image = np.full((n_theta, n_phi), np.nan)
    counts = np.zeros((n_theta, n_phi))

    # Bin indices
    theta_idx = np.digitize(theta, theta_bin_edges, right=True) - 1
    phi_idx = np.digitize(phi, phi_bin_edges, right=True) - 1

    # Only keep valid bins
    valid = (
        (theta_idx >= 0) & (theta_idx < n_theta) &
        (phi_idx >= 0) & (phi_idx < n_phi)
    )

    theta_idx = theta_idx[valid]
    phi_idx = phi_idx[valid]
    dqdx = dqdx[valid]

    # Loop over bins
    for t in range(n_theta):
        #print(t)
        for p in range(n_phi):
            mask = (theta_idx == t) & (phi_idx == p)
            vals = dqdx[mask]

            if len(vals) < min_entries:
                continue

            if statistic == 'mean':
                vals = vals[vals <= 100]
                dqdx_image[t, p] = np.mean(vals)

            elif statistic == 'median':
                dqdx_image[t, p] = np.median(vals)

            elif statistic == 'mpv':
                hist, bins = np.histogram(vals, bins=50, range=(0, 100))
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                dqdx_image[t, p] = bin_centers[np.argmax(hist)]

            else:
                raise ValueError("statistic must be 'mean', 'median', or 'mpv'")

    if maxnorm:
        maxval = np.nanmax(dqdx_image)
        dqdx_image /= maxval

    # Plot
    plt.figure(figsize=(10, 6))
    extent = [
        phi_bin_edges[0], phi_bin_edges[-1],
        theta_bin_edges[0], theta_bin_edges[-1]
    ]

    plt.imshow(
        dqdx_image,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.xlabel("φ [degrees]")
    plt.ylabel("θ [degrees]")
    plt.title(f"{title} ({statistic})")

    cbar = plt.colorbar()
    if maxnorm:
        cbar.set_label("Normalized dQ/dx")
    else:
        cbar.set_label("dQ/dx [ke-/cm]")

    plt.tight_layout()
