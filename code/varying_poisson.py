import sys

import numpy as np

import common
import pickling

alias_varying_poisson = "_varying_poisson"


def main():
    (parameters, folder_path) = common.parse_parameters()
    samples = []
    for varying_poisson_ratio in np.linspace(
        parameters["varying_poisson"][0],
        parameters["varying_poisson"][1],
        parameters["varying_poisson_num_samples"],
    ):
        for varying_poisson_β in np.linspace(
            parameters["varying_poisson_β"][0],
            parameters["varying_poisson_β"][1],
            parameters["varying_poisson_num_samples"],
        ):
            sample = {
                **parameters,
                "poisson_ratio": varying_poisson_ratio,
                "β": varying_poisson_β,
                "φ": parameters["varying_poisson_φ"],
            }
            samples.append(common.compute_sample(sample))
    print(len(samples))
    common.multiprocessing_homogenize(samples)
    pickling.dump_samples(sys.argv[1], samples, prealias=alias_varying_poisson)


if __name__ == "__main__":
    main()
