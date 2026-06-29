import sys

import common
import pickling

alias = "_convergence"
convβid = 0
convφid = 1


def get_conv_tests(params):
    num_points_conv = []
    resolution_conv = []
    resolution_exp, num_points_exp, nk_exp = common.get_params_exp(params)
    samples_half = params["conv_num_points_samples_half"]
    for resolution_exp_test in range(
        resolution_exp - samples_half, resolution_exp + samples_half + 1
    ):
        num_points_exp_test = 2 * resolution_exp_test - nk_exp
        num_points_conv.append(2**num_points_exp_test)
        resolution_conv.append(2**resolution_exp_test)
    return num_points_conv, resolution_conv


def main():
    params, folder = common.parse_parameters()
    samples = []
    for num_points, resolution in zip(*get_conv_tests(params)):
        print("- Num points = " + str(num_points) + "  resolution = " + str(resolution))
        for seed in range(params["conv_random_samples"]):
            sample = {
                **params,
                "num_points": num_points,
                "seed": seed + 1,
            }
            sample["β"] = params["β"][convβid]
            sample["φ"] = params["φ"][convφid]
            sample["resolution"] = resolution
            samples.append(common.compute_sample(sample))

    common.multiprocessing_homogenize(samples)
    pickling.dump_samples(sys.argv[1], samples, prealias=alias)


if __name__ == "__main__":
    main()
