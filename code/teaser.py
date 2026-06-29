import os
import sys

import numpy as np

import common
import curve_interpolation
import halftoned_images_plot
import implicits
import interpolation
import pickling


def get_field_teaser(params, folder_path, compute_field):
    resolution_teaser = 2**14
    num_points_teaser = 2**15

    print(
        "\t- Teaser plot Num points = "
        + str(num_points_teaser)
        + "  resolution = "
        + str(resolution_teaser)
    )
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    data_interp_span = data_interp["E_target_spans"][params["E_target_select_halftonings"]]
    print("Target Young E_t = " + str(data_interp_span["young"]))
    poly_fit_funcs, tex_tau0_dict, tex_poly_dict = curve_interpolation.load_curve_interp(
        folder_path, target_young=data_interp_span["young"]
    )
    print("* Retrieving teaser solid fraction function...")
    image_alias, sigmoid_factor = halftoned_images_plot.image_alias_list[-1]
    func_teaser = halftoned_images_plot.load_image_func(
        os.path.join("data", image_alias + ".png"),
        data_interp_span["solid_fraction_range"],
        sigmoid_factor,
    )
    if not compute_field:
        return func_teaser
    print("* Computing teaser field...")
    field_teaser = implicits.triangulation(
        n=params["n"],
        d=params["d"],
        φ=None,
        β=None,
        resolution=resolution_teaser,
        seed=params["seed"],
        num_points=num_points_teaser,
        upsampling=3,
        func_target_solid_fraction=func_teaser,
        func_poly_φ=poly_fit_funcs["φ"],
        func_poly_β=poly_fit_funcs["β"],
    )
    return field_teaser, func_teaser


def main():
    params, folder_path = common.parse_parameters()
    field_teaser, func_teaser = get_field_teaser(params, folder_path, compute_field=True)
    field_teaser_pad = np.pad(field_teaser, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    filename_image = os.path.join(folder_path, "teaser.png")
    halftoned_images_plot.save_halftoning_image(
        field_teaser_pad,
        filename_image,
    )


if __name__ == "__main__":
    main()
