import sys

import numpy as np

import common
import pickling

alias = "_convergence_pixels_per_point"


def main():
    params, folder_path = common.parse_parameters()
    resolution_exp, num_points_exp, nk_exp = common.get_params_exp(params)
    ppp_half_sample = params["ppp_num_samples_half"]
    resolutions_exp = np.arange(
        resolution_exp - ppp_half_sample,
        resolution_exp + ppp_half_sample + 1,
        1,
    )
    samples = []
    ppp_id = 0
    print("* Resolution power of two exponents to test = " + str(resolutions_exp))
    for β in params["β"]:
        for φ in params["φ"]:
            print(f"* Case β = {β}   φ = {φ}")
            for resolution_exp in resolutions_exp:
                print(
                    "\t- Resolution = 2^"
                    + str(resolution_exp)
                    + "   Num points = 2^"
                    + str(num_points_exp)
                    + "   n_k = 2^"
                    + str(2 * resolution_exp - num_points_exp)
                )
                for i in range(params["ppp_num_random_realizations"]):
                    sample = {
                        **params,
                        "num_points": params["num_points"],
                        "β": β,
                        "φ": φ,
                        "resolution": 2**resolution_exp,
                        "seed": params["seed"] + i,
                        "ppp_id": ppp_id,
                    }
                    samples.append(common.compute_sample(sample))
            ppp_id += 1
    pickling.dump_samples(sys.argv[1], samples, prealias=alias)


if __name__ == "__main__":
    main()
