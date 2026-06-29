import sys

import numpy as np

import common
import pickling

if __name__ == "__main__":
    assert len(sys.argv) == 2
    parameters = common.load_parameters(sys.argv[1])
    constant_parameters = []
    varying_parameters = []
    varying_parameters_bounds = []
    for key in parameters:
        value = parameters[key]
        if isinstance(value, list) and (key == "β" or key == "φ"):
            assert all(isinstance(elem, (int, float)) for elem in value)
            assert len(value) == 2
            assert value[0] < value[1]
            varying_parameters.append(key)
            varying_parameters_bounds.append(value)
        else:
            constant_parameters.append(key)
    if len(varying_parameters) > 0:
        print("* Sampling space of varying parameters")
        sampling = [
            np.linspace(bound[0], bound[1], num=parameters["num_samples_param"])
            for bound in varying_parameters_bounds
        ]
        flattened_sampling = [grid.ravel() for grid in np.meshgrid(*sampling)]
        samples_varying_parameters = [element for element in zip(*flattened_sampling)]
    else:
        samples_varying_parameters = [parameters]

    # compute samples (serial)
    samples = []
    for sample_varying_parameters in samples_varying_parameters:
        print("\n* Computing sample {0}".format(len(samples)))
        sample = {}
        for constant_parameter in constant_parameters:
            sample[constant_parameter] = parameters[constant_parameter]
        for param_i in range(len(varying_parameters)):
            varying_parameter_i = varying_parameters[param_i]
            sample[varying_parameter_i] = sample_varying_parameters[param_i]
        print(sample)
        samples.append(common.compute_sample_realizations(sample))

    common.multiprocessing_homogenize(samples, fun=common.homogenize_sample_realizations)
    pickling.dump_samples(sys.argv[1], samples)
