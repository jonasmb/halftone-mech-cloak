import functools
import json
import os

import numpy as np
import scipy

alias_curve_interpolation = "curve_interp"


def polynomial_fitting(x, y, degree):
    xspan = x[-1] - x[0]
    assert xspan > 0

    def fun(xc, *coeffs):
        x0 = (xc - x[0]) / xspan
        result = y[0]
        for i, c in enumerate(coeffs):
            result += c * x0 ** (i + 1)
        return result

    sigma = np.ones(len(x) - 1)
    # give the last point prominence in the fitting, the first one is fixed
    sigma[-1] = 1e-4
    p0 = np.ones(degree)
    popt, pcov = scipy.optimize.curve_fit(
        f=fun, xdata=x[1:], ydata=y[1:], sigma=sigma, ftol=1.0e-14, p0=p0
    )
    print(
        "scipy.optimize.curve_fit -> estimated approximate covariance of popt (pcov) =" + str(pcov)
    )

    def polyt_fit_func(xc):
        return fun(xc, *popt)

    # check endpoints
    print(
        "* samples vs curve fit (y deviation, should be small) = ["
        + str([y[0] - polyt_fit_func(x[0]), y[-1] - polyt_fit_func(x[-1])])
    )
    return popt


def curve_fitting_βφ(data_interp, elem):
    data_curve_fit = {"β": {"degree": 8}, "φ": {"degree": 5}}
    data_curve_fit["E_target"] = data_interp["E_target_spans"][elem]["young"]
    for i, prop in enumerate(["β", "φ"]):
        x, y = (
            data_interp["E_target_spans"][elem]["solid_fraction"],
            data_interp["E_target_spans"][elem]["vertices"][:, i],
        )

        if np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0):
            print("Samples of " + prop + " are strictly monotonic")
        x = np.flip(x)
        y = np.flip(y)
        popt = polynomial_fitting(x, y, data_curve_fit[prop]["degree"])
        data_curve_fit[prop]["xini"] = str(x[0])
        data_curve_fit[prop]["xend"] = str(x[-1])
        data_curve_fit[prop]["yini"] = str(y[0])
        data_curve_fit[prop]["popt"] = np.array2string(popt, max_line_width=1e6, precision=20)
        # output to tex
        precision = 2
        inirs = str(np.round(x[0], precision))
        endrs = str(np.round(x[-1], precision))
        tex_tau0 = r"\tau \in [" + inirs + "," + endrs + "],"
        tex_tau0 += r"\quad \tau_{0} = \frac{\tau - " + inirs + "}{" + endrs + "-" + inirs + "}"
        tex_poly = ""
        if y[0] != 0:
            tex_poly += str(np.round(y[0], precision))
        max_coeff_per_line = 6
        num_coeff = 0
        for i, c in enumerate(popt):
            if c != 0:
                if c > 0 and i > 0:
                    tex_poly += "+"
                tex_poly += str(np.round(c, precision)) + r"\tau_{0}"
                if i > 0:
                    tex_poly += "^" + str(i + 1)
                num_coeff += 1
                if num_coeff >= max_coeff_per_line and i < len(popt) - 1:
                    num_coeff = 0
                    tex_poly += r"& \\ &"

        tex_poly += r"\quad \in [{0}, {1}]".format(
            *np.round([min(y[0], y[-1]), max(y[0], y[-1])], precision)
        )
        data_curve_fit[prop]["tex_tau0"] = tex_tau0
        data_curve_fit[prop]["tex_poly"] = tex_poly
    return data_curve_fit


def to_percent(a):
    assert (a * 100).is_integer()
    return int(a * 100)


def target_young_percent(target_young):
    return "_targetyoung_" + str(to_percent(target_young)) + "_percent"


def filename_percent(folder_path, target_young):
    return os.path.join(
        folder_path,
        alias_curve_interpolation + target_young_percent(target_young),
    )


def filename_json(folder_path, target_young):
    return filename_percent(folder_path, target_young) + ".json"


def save_curve_interp(folder_path, data_curve_interp):
    with open(filename_json(folder_path, data_curve_interp["E_target"]), "w") as outfile:
        json.dump(data_curve_interp, outfile)


def poly_fun(xc, xini, xend, yini, popt):
    x0 = (xc - xini) / (xend - xini)
    result = yini
    for i, c in enumerate(popt):
        result += c * x0 ** (i + 1)
    return result


def load_data_curve_fit(folder_path, target_young):
    data_curve_fit = None
    with open(filename_json(folder_path, target_young), "r") as infile:
        data_curve_fit = json.load(infile)
    assert data_curve_fit is not None
    return data_curve_fit


def load_curve_interp(folder_path, target_young):
    data_curve_fit = load_data_curve_fit(folder_path, target_young)
    poly_fit_funcs = {}
    tex_tau0_dict = {}
    tex_poly_dict = {}
    for prop in ["β", "φ"]:
        poly_fit_funcs[prop] = None
        tex_tau0_dict[prop] = data_curve_fit[prop]["tex_tau0"]
        tex_poly_dict[prop] = data_curve_fit[prop]["tex_poly"]
        popt = np.fromstring(data_curve_fit[prop]["popt"].strip("[]"), sep=" ", dtype=np.float64)
        print(prop + " popt = " + str(popt))
        poly_fit_funcs[prop] = functools.partial(
            poly_fun,
            xini=np.float64(data_curve_fit[prop]["xini"]),
            xend=np.float64(data_curve_fit[prop]["xend"]),
            yini=np.float64(data_curve_fit[prop]["yini"]),
            popt=popt,
        )
    return poly_fit_funcs, tex_tau0_dict, tex_poly_dict
