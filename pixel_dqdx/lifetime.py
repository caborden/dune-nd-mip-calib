import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def log_with_error(values, errors):
    log_val = np.log(values)
    log_err = []
    for val, err in zip(values, errors):
        if val <= 0:
            raise ValueError("Cannot take log of negative value")
        log_err.append(err / val)
    return log_val, np.array(log_err)

def lifetime_fit(results_tpc0, runtype, bool_plot=False):
    xn = results_tpc0['bin'][~np.isnan(results_tpc0['bin'])]
    xn = xn[::-1] * 0.1  # Convert to μs and flip

    y = np.array(results_tpc0['mpv'])
    e = np.array(results_tpc0['mpv_error'])

    y = y[~np.isnan(y)][::-1]
    e = e[~np.isnan(e)][::-1]

    y, e = log_with_error(y, e)
    #emask = e < 0.005
    emask = slice(6,-3)
    #emask = slice(3,-3)

    X_fit = xn[emask]
    y_fit = y[emask]
    e_fit = e[emask]

    def linear_model(x, m, b):
        return m * x + b

    def diffusion_model(x, m, b, alpha=1.1):
        diff_term = alpha * np.log(1 + (2 * 6.34e-6 * x) / (0.37**2))
        lifetime_term = linear_model(x,m,b)
        return lifetime_term + diff_term

    lower_bounds = [-np.inf, 1.0, 1]
    upper_bounds = [np.inf, 10.0, 5]

    popt, pcov = curve_fit(linear_model, X_fit, y_fit,
                           sigma=e_fit, absolute_sigma=True,
                           p0=[-0.0009, 40.11])

    m, b = popt
    sigma_m = np.sqrt(pcov[0, 0]) # Standard error of the slope

    lifetime = -1.0 / m
    lifetime_error = (1.0 / (m**2)) * sigma_m  # (lifetime**2) * sigma_m

    print(type(lifetime), np.shape(lifetime))
    print(type(lifetime_error), np.shape(lifetime_error))

    y_pred = linear_model(X_fit, m, b)
    chi2 = np.sum(np.square(y_pred - y_fit) / np.square(e_fit))
    ndf = len(X_fit) - 2

    # plotting
    if bool_plot:
        plt.errorbar(X_fit, y_fit, yerr=e_fit, color='#444444', label='Data', ls='none', fmt='.', capsize=3)
        plt.plot(X_fit, y_pred, color='coral', linewidth=1.5, label='Fit')

        plt.plot([], [], ' ', label=fr"Lifetime: {lifetime:.2f} $\pm$ {lifetime_error:.2f} $\mu$s")
        #plt.plot([], [], ' ', label=f"Diff. alpha: {1.1:.2f}")

        plt.title(fr'$\chi^2$/ndf = {chi2/ndf:.2f}')
        plt.xlabel(r'Drift Time ($\mu$s)')
        plt.ylabel('ln(dQ/dx)')
        plt.legend()
        plt.grid(True)
        plt.savefig('lifetime_fit_'+str(runtype)+'.pdf')
        plt.show()

    return lifetime, lifetime_error
