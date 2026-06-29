import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy

import common
import curve_interpolation
import implicits
import interpolation
import pickling
import plots

image_alias_list = [
    ("hokusai", 10),
    ("foam", None),
    ("example_qr", None),
    ("floral_design", 10),
]


def save_halftoning_image(field_halftoning, filename):
    field_halftoning_im = PIL.Image.fromarray((field_halftoning * 255).astype(np.uint8))
    field_halftoning_im.save(filename)


def load_image_func(filename, max_span_range_solid_fraction_young, sigmoid_factor):
    image = PIL.Image.open(filename)
    image_data = np.array(np.flipud(image)).T / 255.0
    assert len(image_data.shape) == 2
    assert image_data.shape[0] == image_data.shape[1]
    resolution = image_data.shape[0]
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    image_points = (
        np.linspace(0, 1, resolution),
        np.linspace(0, 1, resolution),
    )
    image_interpolator = scipy.interpolate.RegularGridInterpolator(
        image_points,
        image_data,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    def func_target_solid_fraction_image(
        p,
    ):
        interp = image_interpolator((p[0], p[1]))
        if sigmoid_factor is not None:
            trans_func = 1 - 1 / (1 + np.exp(-sigmoid_factor * (interp - 0.5)))
        else:
            trans_func = 1 - interp
        return max_span_range_solid_fraction_young[0] + trans_func * (
            max_span_range_solid_fraction_young[1] - max_span_range_solid_fraction_young[0]
        )

    return func_target_solid_fraction_image


def load_funcs_target_solid_fraction_paper(data_interp_span):
    solid_fraction_func_dict = {}

    def func_target_solid_fraction_two_phases(p):
        return np.where(
            p[0] < 0.5,
            data_interp_span["solid_fraction_range"][0],
            data_interp_span["solid_fraction_range"][1],
        )

    solid_fraction_func_dict["two_phases"] = func_target_solid_fraction_two_phases

    for image_alias, sigmoid_factor in image_alias_list:
        filename = os.path.join("data", image_alias + ".png")
        image_func = load_image_func(
            filename,
            data_interp_span["solid_fraction_range"],
            sigmoid_factor,
        )
        solid_fraction_func_dict[image_alias] = image_func

    def func_target_solid_fraction_julia(p):
        center = -0.01 - 0.72j
        range_ = 1.05 + abs(center)
        max_iterations = 256
        z = p[0] + 1j * p[1]
        iteration = 0
        while iteration < max_iterations:
            z = np.where(np.abs(z) < range_, z**2 + center, z)
            iteration += 1
        z = np.where(np.abs(z) < 2.4, 0, 1)
        return data_interp_span["solid_fraction_range"][0] + z * (
            data_interp_span["solid_fraction_range"][1]
            - data_interp_span["solid_fraction_range"][0]
        )

    solid_fraction_func_dict["julia_set"] = func_target_solid_fraction_julia
    return solid_fraction_func_dict


def compute_poissons_ratio_deviation(
    X, data, data_interp, poly_fit_funcs, func_target_solid_fraction
):
    # Poisson's ratio deviation error
    βX = poly_fit_funcs["β"](func_target_solid_fraction(X))
    φX = poly_fit_funcs["φ"](func_target_solid_fraction(X))
    interp_xi = np.stack([βX, φX], axis=-1)
    interp_xi = np.reshape(interp_xi, (plots.subdiv_val * plots.subdiv_val, 2))
    interp_poisson = scipy.interpolate.griddata(
        points=data_interp["interp_points"],
        values=data["poisson_iso"],
        xi=interp_xi,
        method=interpolation.interp_method,
    )
    interp_poisson = np.reshape(interp_poisson, (plots.subdiv_val, plots.subdiv_val))
    assert not np.any(np.isnan(interp_poisson))
    return interp_poisson


def main():
    params, folder_path = common.parse_parameters()
    resolution_halftonings = 4096
    folder_path_halfoned_images = os.path.join(folder_path, "halftoned_images")
    data = common.load_data(sys.argv[1])
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    data_interp_span = data_interp["E_target_spans"][params["E_target_select_halftonings"]]
    print("Target Young E_t = " + str(data_interp_span["young"]))
    if not os.path.exists(folder_path_halfoned_images):
        os.makedirs(folder_path_halfoned_images)
    ncols = 7
    poly_fit_funcs, tex_tau0_dict, tex_poly_dict = curve_interpolation.load_curve_interp(
        folder_path, target_young=data_interp_span["young"]
    )

    def plot_halftoning(ax, func_target_solid_fraction, plot_titles, alias):
        num_results = len(ax) - 2
        X = np.meshgrid(
            np.linspace(0, 1, plots.subdiv_val),
            np.linspace(0, 1, plots.subdiv_val),
        )
        plots.set_rasterized(
            ax[num_results].contourf(
                X[0],
                X[1],
                func_target_solid_fraction(X),
                vmin=0,
                vmax=1,
                levels=plots.subdiv_val,
                cmap=plots.cmap_settings["mat"],
            )
        )
        if plot_titles:
            ax[num_results].set_title(plots.labels["target_solid_fraction"])
            ax[ncols - 1].set_title("Poisson's ratio")

        def default(ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        for i in range(len(ax)):
            default(ax[i])
        interp_poisson = compute_poissons_ratio_deviation(
            X, data, data_interp, poly_fit_funcs, func_target_solid_fraction
        )
        contour_poisson = ax[ncols - 1].contourf(
            X[0],
            X[1],
            interp_poisson,
            cmap=plots.cmap_settings["poisson_deviation"],
            levels=plots.subdiv_val,
        )
        fig.colorbar(
            contour_poisson,
            ax=ax[ncols - 1],
            orientation="vertical",
            label=r"$v$",
            ticks=np.linspace(
                np.min(interp_poisson),
                np.max(interp_poisson),
                plots.global_num_ticks_cb,
            ),
            shrink=0.85,
        )
        interp_poisson_mean = np.mean(interp_poisson)
        print("\t\t- Average Poisson's ratio field = " + str(interp_poisson_mean))
        ax[ncols - 1].text(
            0.1,
            -0.1,
            r"Mean $v \approx" + str(np.round(interp_poisson_mean, 5)) + "$",
            va="center",
        )
        plots.set_rasterized(contour_poisson)

        # results with increasing number of points
        min_power_two, off_power_two = 4, 9
        for i in range(num_results):
            num_points_halftoning = 2 ** (
                min_power_two + int(np.round(i / float(num_results - 1) * off_power_two))
            )
            print("\t\t- num_points_halftoning = " + str(num_points_halftoning))
            field_halftoning = implicits.triangulation(
                n=params["n"],
                d=params["d"],
                φ=None,
                β=None,
                resolution=resolution_halftonings,
                seed=params["seed"],
                num_points=num_points_halftoning,
                upsampling=params["upsampling"],
                func_target_solid_fraction=func_target_solid_fraction,
                func_poly_φ=poly_fit_funcs["φ"],
                func_poly_β=poly_fit_funcs["β"],
            )
            save_halftoning_image(
                field_halftoning,
                os.path.join(
                    folder_path_halfoned_images,
                    alias + "num_points_" + str(num_points_halftoning) + ".png",
                ),
            )
            ax[i].imshow(
                field_halftoning,
                vmin=0,
                vmax=1,
                cmap=plots.cmap_settings["mat"],
                extent=(0, resolution_halftonings, 0, resolution_halftonings),
                interpolation=plots.imshowinterp,
            )
            if plot_titles:
                ax[i].set_title("Points = " + str(num_points_halftoning))

    solid_fraction_func_dict = load_funcs_target_solid_fraction_paper(
        data_interp_span=data_interp_span
    )

    fig, axes = plots.plot_init(
        to_pgf=True, wide=True, ncols=ncols, nrows=len(solid_fraction_func_dict)
    )
    i = 0
    for alias, solid_fraction_func in solid_fraction_func_dict.items():
        print("\t- " + alias)
        plot_halftoning(axes[i], solid_fraction_func, plot_titles=(i == 0), alias=alias)
        i += 1
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0.2)
    plots.save_pgf(folder_path, "halftonings")


if __name__ == "__main__":
    main()
