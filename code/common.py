import copy
import json
import multiprocessing
import os
import sys

import numpy as np
import scipy

import homogenization
import implicits
import isotropy
import pickling
import plots


def decimal_to_latex_mantissa(value, rnd):
    position = -int(np.floor(np.log10(abs(value))))
    mantissa = value * (10**position)
    return f"{mantissa:.{rnd}f} \\times 10^{{-{position}}}"


def ensure_int(n):
    return int(n) if isinstance(n, (int, np.integer)) else None


def get_exp_power_two(n_in):
    n = ensure_int(n_in)
    if n is None:
        return None
    if n > 0 and (n & (n - 1)) == 0:  # if is power of two
        return n.bit_length() - 1
    return None


def get_params_exp(params):
    resolution_exp = get_exp_power_two(int(params["resolution"]))
    assert resolution_exp is not None
    num_points_exp = get_exp_power_two(int(params["num_points"]))
    assert num_points_exp is not None
    nk_exp = resolution_exp * 2 - num_points_exp
    print(
        "* Resolution = 2^"
        + str(resolution_exp)
        + "   Num points = 2^"
        + str(num_points_exp)
        + "   Target n_k = 2^"
        + str(nk_exp)
    )
    return resolution_exp, num_points_exp, nk_exp


def init_folder_path(filename):
    folder = os.path.splitext(filename)[0]
    os.makedirs(folder, exist_ok=True)
    return folder


def load_parameters(filename_json):
    """
    Load the parameters from a json file.

    Args:
        filename_json: filename of the input json file.

    Returns:
        Dictionary of parameters.
    """
    with open(filename_json, "r") as f:
        params = json.load(f)
    print("*** Parameters", json.dumps(params, indent=4))
    return params


def parse_parameters():
    assert len(sys.argv) >= 2
    return load_parameters(sys.argv[1]), init_folder_path(sys.argv[1])


def write_value_tex(f, alias, value):
    alias = (
        alias.replace("_", "")
        .replace("β", "beta")
        .replace("φ", "varphi")
        .replace("10", "ten")
        .replace("20", "twenty")
        .replace("30", "thirty")
    )
    f.write(f"\\newcommand{{\\{alias}val}}{{{value}}}\n")


def write_value_tex_and_dict(f, alias, value, d):
    write_value_tex(f, alias, value)
    d[alias] = value


def num_connected_solid_phases(image):
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return np.count_nonzero(
        np.unique(
            scipy.ndimage.label(input=np.tile(image, (3, 3)), structure=structure)[0][
                image.shape[0] : image.shape[0] * 2,
                image.shape[1] : image.shape[1] * 2,
            ]
        )
    )


def homogenize_sample(sample):
    """
    Homogenization of a sample.

    We obtain the distribution of Young moduli by transforming
    the field [0,1] to [0, E_max]

    Args:
        sample: input sample.

    Returns:
        Homogenized elasticity/stiffness tensor.
    """
    assert num_connected_solid_phases(sample["field"]) == 1

    E_field = isotropy.E_HS_up(
        tau=sample["field"],
        E_s=sample["young_max"],
    )
    ν = sample["poisson_ratio"]
    E_factor = (1 + 2 * ν) / (1 + ν) ** 2
    ls_op = homogenization.LippmannSchwingerOperator(
        E=E_factor * E_field,
        E0=1.1 * E_factor * sample["young_max"],
        ν=ν / (1 + ν),
    )
    C_eff = homogenization.effective_stiffness(ls_op, tol=sample["tol"], useUmfpack=True)
    print(f"\n{sample}\n*C_eff=\n{C_eff}")
    return C_eff


def compute_sample(sample):
    """
    Computation of sample image.

    Given an input sample description, compute a discrete image.

    Args:
        sample: input sample description.

    Returns:
        Gray-scale 2D image.
    """
    sample["field"] = implicits.triangulation(
        n=sample["n"],
        d=sample["d"],
        φ=sample["φ"],
        β=sample["β"],
        resolution=sample["resolution"],
        seed=sample["seed"],
        num_points=sample["num_points"],
        upsampling=sample["upsampling"],
    )
    return sample


def multiprocessing_homogenize(samples, fun=homogenize_sample):
    """
    Homogenization of samples in parallel.

    Given a list of sample descriptions, homogenize each one in parallel.

    Args:
        samples: list of sample descriptions.
    """
    with multiprocessing.Pool(max(multiprocessing.cpu_count() - 3, 1)) as pool:
        results = pool.starmap(fun, [(s,) for s in samples])
    for i, s in enumerate(samples):
        s.update({"C_eff": results[i], "num_sample": i})
    return samples


def compute_sample_realizations(sample):
    fields = []
    for i in range(sample["conv_random_samples"]):
        print("* Random realization i = " + str(i))
        realization_i = copy.copy(sample)
        realization_i["seed"] = sample["seed"] + i
        field_i = compute_sample(realization_i)
        fields.append(field_i["field"])
    sample["field"] = fields
    return sample


def homogenize_sample_realizations(sample):
    C_eff_realizations = []
    for i in range(len(sample["field"])):
        print("* Homogenization of random realization i = " + str(i))
        realization_i = copy.copy(sample)
        realization_i["seed"] = sample["seed"] + i
        realization_i["field"] = sample["field"][i]
        C_eff_i = homogenize_sample(realization_i)
        C_eff_realizations.append(C_eff_i)
    print("* Collected Ceff of realizations")
    print(C_eff_realizations)
    print("* Mean of collected Ceff")
    print(np.mean(C_eff_realizations, axis=0))
    return C_eff_realizations


def load_data(filename, prealias=""):
    print(f"* Loading data from {filename}")
    samples = pickling.load_samples(filename, prealias)
    data = {
        "bulk_iso": [],
        "shear_iso": [],
        "dev_iso": [],
        "young_iso": [],
        "poisson_iso": [],
        "solid_fraction": [],
    }
    for s in samples:
        for sk in s:
            if sk not in data:
                data[sk] = []
            data[sk].append(s[sk])
        solid_fraction = np.mean(s["field"])
        data["solid_fraction"].append(solid_fraction)
        if "C_eff" in s:
            if isinstance(s["C_eff"], np.ndarray):
                C_eff = s["C_eff"]  # one realization
            else:
                C_eff = np.mean(s["C_eff"], axis=0)  # multiple realizations
            bulk, shear, dev = isotropy.closest_isotropic(C_eff)
            data["bulk_iso"].append(bulk)
            data["shear_iso"].append(shear)
            data["dev_iso"].append(dev)
            data["young_iso"].append(isotropy.E(bulk, shear))
            data["poisson_iso"].append(isotropy.v(bulk, shear))
    return data


def plot_hs_upper_bound(ax, name, params):
    tau = np.linspace(0, 1, 128)
    val = {
        "bulk_iso": isotropy.κ_HS_up(tau=tau, E_s=params["young_max"], v_s=params["poisson_ratio"]),
        "shear_iso": isotropy.μ_HS_up(
            tau=tau, E_s=params["young_max"], v_s=params["poisson_ratio"]
        ),
        "young_iso": isotropy.E_HS_up(tau=tau, E_s=params["young_max"]),
    }.get(name)
    if val is not None:
        ax.plot(
            tau,
            val,
            lw=0.5,
            c="tab:brown",
            label=plots.labels[f"{name}_hs_up"],
        )
        ax.legend()
