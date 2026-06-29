import sys

import numpy as np

import common
import implicits
import pickling

alias = "_mean_fluctuations"


def get_stats(X):
    mean_X = np.mean(X)
    print("\tMean = " + str(mean_X))
    std_X = np.std(X)
    print("\tStandard Deviation = " + str(std_X))
    coeff_var_X = std_X / mean_X
    print("\tCoefficient of Variation = " + str(std_X / mean_X))
    return mean_X, std_X, coeff_var_X


def main():
    (params, folder_path) = common.parse_parameters()
    tri_thick_means = []
    tri_thick_fields = []
    assert params["mean_fluc_num_samples"] <= params["mean_fluc_num_samples"]
    mean_fluc_thick_factor = 0.0222  # arbitrary factor to have close to 0.5 solid fraction
    print("* Computing samples triangulation with thickened edges...")
    mean_fluc_tol_mean_diff = 0.001
    mean_fluc_max_iter_bin_search = 64
    mean_fluc_φ = (params["φ"][0] + params["φ"][1]) / 2
    for seed_off in range(params["mean_fluc_num_samples"]):
        print("\t- Sample " + str(seed_off))
        field_tri_thick = implicits.triangulation(
            n=None,
            d=params["mean_fluc_resolution"]
            * mean_fluc_thick_factor,  # this field is (over)used for the thickness value
            φ=None,
            β=None,
            resolution=params["mean_fluc_resolution"],
            seed=params["seed"] + seed_off,
            num_points=params["num_points"],
            upsampling=params["upsampling"],
            triangle_function=implicits.dist_to_tri_edges,
        )
        tri_thick_means.append(np.mean(field_tri_thick))
        tri_thick_fields.append(field_tri_thick)
    tri_thick_mean = np.mean(tri_thick_means)

    print("* Binary search to approximatively match solid fractions...")
    # binary search in order to find a parameter beta
    # that closely approximates the mean solid fraction of
    # the method with thickened edges
    low_β, high_β = params["β"][0], params["β"][1]
    mid_β = None
    i = 0
    while i < mean_fluc_max_iter_bin_search:
        mid_β = (low_β + high_β) / 2
        field_S = implicits.triangulation(
            n=params["n"],
            d=params["d"],
            φ=mean_fluc_φ,
            β=mid_β,
            resolution=params["mean_fluc_resolution"],
            seed=params["seed"] + seed_off,
            num_points=params["num_points"],
            upsampling=params["upsampling"],
        )
        field_S_mean = np.mean(field_S)
        if np.abs(field_S_mean - tri_thick_mean) <= mean_fluc_tol_mean_diff:
            break
        if field_S_mean > tri_thick_mean:
            high_β = mid_β
        else:
            low_β = mid_β
        i += 1
    assert mid_β is not None
    mean_fluc_β = mid_β
    print("* Computing samples triangulation with our method...")
    S_means = []
    S_fields = []
    for seed_off in range(params["mean_fluc_num_samples"]):
        print("\t- Sample " + str(seed_off))
        field_S = implicits.triangulation(
            n=params["n"],
            d=params["d"],
            φ=mean_fluc_φ,
            β=mean_fluc_β,
            resolution=params["mean_fluc_resolution"],
            seed=params["seed"] + seed_off,
            num_points=params["num_points"],
            upsampling=params["upsampling"],
        )
        S_means.append(np.mean(field_S))
        S_fields.append(field_S)

    print("* Thickening triangles function")
    get_stats(tri_thick_means)
    print("* S function (our)")
    get_stats(S_means)
    results = {
        "tri_thick_means": tri_thick_means,
        "tri_thick_fields": tri_thick_fields,
        "S_means": S_means,
        "S_fields": S_fields,
    }
    pickling.dump_samples(sys.argv[1], results, prealias=alias)


if __name__ == "__main__":
    main()
