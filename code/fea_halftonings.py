import gc
import os
import sys

import numpy as np

import common
import convergence
import curve_interpolation
import fea_validation
import halftoned_images_plot
import implicits
import interpolation
import isotropy
import pickling
import plots

alias_fea_halftonings = "fea_halftonings"


def get_fea_result(params, field_solid_fraction, fea_data_dict):
    field_young_moduli = isotropy.E_HS_up(
        tau=field_solid_fraction,
        E_s=params["young_max"],
    )
    fea_data = {
        "field_solid_fraction": field_solid_fraction,
        "field_young_moduli": field_young_moduli,
        "fea_results": fea_validation.precompute_fea(
            x=np.flipud(field_young_moduli),
            E_s=fea_data_dict["young_solid"],
            v_s=fea_data_dict["poisson_solid"],
            E_t=fea_data_dict["young_target"],
            v_t=fea_data_dict["poisson_solid"],
        ),
    }
    return fea_data


def main():
    params, folder_path = common.parse_parameters()
    conv_tests = convergence.get_conv_tests(params)
    """
    cv1, cv2 = -1, 2
    conv_tests = [
        (conv_tests_all[0][cv1], conv_tests_all[0][cv2]),
        (conv_tests_all[1][cv1], conv_tests_all[1][cv2]),
    ]"""

    # TEMP experiment with little size
    """cv1, cv2 = 0, 1
    conv_tests = [
        (conv_tests[0][cv1], conv_tests[0][cv2]),
        (conv_tests[1][cv1], conv_tests[1][cv2]),
    ]"""

    data = common.load_data(sys.argv[1])
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    data_interp_span = data_interp["E_target_spans"][params["E_target_select_halftonings"]]
    print("Target Young E_t = " + str(data_interp_span["young"]))
    folder_path_halfoned_images = os.path.join(folder_path, "halftoned_images_fea")
    if not os.path.exists(folder_path_halfoned_images):
        os.makedirs(folder_path_halfoned_images)
    poly_fit_funcs, tex_tau0_dict, tex_poly_dict = curve_interpolation.load_curve_interp(
        folder_path, target_young=data_interp_span["young"]
    )

    solid_fraction_func_dict = halftoned_images_plot.load_funcs_target_solid_fraction_paper(
        data_interp_span
    )
    X = np.meshgrid(np.linspace(0, 1, plots.subdiv_val), np.linspace(0, 1, plots.subdiv_val))

    fea_data_dict = {
        "young_solid": params["young_max"],
        "poisson_solid": params["poisson_ratio"],
        "young_target": float(data_interp_span["young"]),
        "tests": {},
    }
    print("* Precompute FEA gallery halftoning...")
    i = 0
    for alias, solid_fraction_func in solid_fraction_func_dict.items():
        print("* " + alias)
        print("\t- Young's modulus target = " + str(fea_data_dict["young_target"]))
        fea_data_alias_dict = {}
        for num_points, resolution in zip(*conv_tests):
            print("\t- Num points = " + str(num_points) + "  resolution = " + str(resolution))
            field_solid_fraction = implicits.triangulation(
                n=params["n"],
                d=params["d"],
                φ=None,
                β=None,
                resolution=resolution,
                seed=params["seed"],
                num_points=num_points,
                upsampling=params["upsampling"],
                func_target_solid_fraction=solid_fraction_func,
                func_poly_φ=poly_fit_funcs["φ"],
                func_poly_β=poly_fit_funcs["β"],
            )
            halftoned_images_plot.save_halftoning_image(
                field_solid_fraction,
                os.path.join(
                    folder_path_halfoned_images,
                    alias
                    + "num_points_"
                    + str(num_points)
                    + "_resolution"
                    + str(resolution)
                    + ".png",
                ),
            )
            fea_data = get_fea_result(
                params,
                field_solid_fraction,
                fea_data_dict,
            )

            # compute error between heterogeneous and homogeneus result (L_2 norm)
            for i in range(len(fea_data["fea_results"])):
                mode = fea_data["fea_results"][i]["mode"]
                U = fea_data["fea_results"][i]["nodal_displacements"]
                U_t = fea_data["fea_results"][i]["homogeneous_nodal_displacements"]
                disp_error = np.sqrt(np.sum((U - U_t) ** 2)) / np.sqrt(np.sum(U_t**2))
                fea_data["fea_results"][i]["disp_error"] = disp_error
                print(str(mode) + "disp_error=" + str(disp_error))

            fea_data_alias_dict[(num_points, resolution)] = fea_data

            gc.collect()
        fea_data_dict["tests"][alias] = fea_data_alias_dict
        i += 1
    pickling.dump_samples(
        sys.argv[1],
        fea_data_dict,
        prealias=alias_fea_halftonings,
    )


if __name__ == "__main__":
    main()
