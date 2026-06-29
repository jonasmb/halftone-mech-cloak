import sys
import os
import numpy as np
import common
import convergence
import pickling
import interpolation
import halftoned_images_plot
import curve_interpolation
import implicits
import isotropy

alias_periodic_test = "periodic_test"
folder_path_halfoned_images = "periodic_test"


def main():
    params, folder_path = common.parse_parameters()
    folder_path_halfoned_images = os.path.join(folder_path, "halftoned_images_periodic_test")
    if not os.path.exists(folder_path_halfoned_images):
        os.makedirs(folder_path_halfoned_images)
    conv_tests = convergence.get_conv_tests(params)
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    samples = []
    for elem in range(len(data_interp["E_target_spans"])):
        data_interp_span = data_interp["E_target_spans"][elem]
        print("Target Young E_t = " + str(data_interp_span["young"]))
        poly_fit_funcs, tex_tau0_dict, tex_poly_dict = curve_interpolation.load_curve_interp(
            folder_path, target_young=data_interp_span["young"]
        )

        def func_target_solid_fraction_periodic_square(p):
            t = np.sin(np.pi * p[0]) ** 4 * np.sin(np.pi * p[1]) ** 4
            return (1 - t) * data_interp_span["solid_fraction_range"][0] + t * data_interp_span[
                "solid_fraction_range"
            ][1]

        for num_points, resolution in zip(*conv_tests):
            print("\t- Num points = " + str(num_points) + "  resolution = " + str(resolution))
            sample = {
                "young_max": params["young_max"],
                "poisson_ratio": params["poisson_ratio"],
                "tol": params["tol"],
                "n": params["n"],
                "d": params["d"],
                "resolution": resolution,
                "num_points": num_points,
                "upsampling": params["upsampling"],
                "target_young": data_interp_span["young"],
            }
            sample["field"] = implicits.triangulation(
                n=sample["n"],
                d=sample["d"],
                φ=None,
                β=None,
                resolution=resolution,
                seed=params["seed"],
                num_points=num_points,
                upsampling=sample["upsampling"],
                func_target_solid_fraction=func_target_solid_fraction_periodic_square,
                func_poly_φ=poly_fit_funcs["φ"],
                func_poly_β=poly_fit_funcs["β"],
                int_tri_eval_outside=True,
            )
            samples.append(sample)
            halftoned_images_plot.save_halftoning_image(
                sample["field"],
                os.path.join(
                    folder_path_halfoned_images,
                    str(elem)
                    + "num_points_"
                    + str(num_points)
                    + "_resolution"
                    + str(resolution)
                    + ".png",
                ),
            )
    common.multiprocessing_homogenize(samples)
    pickling.dump_samples(sys.argv[1], samples, prealias=alias_periodic_test)


if __name__ == "__main__":
    main()
