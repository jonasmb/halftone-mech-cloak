import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy

import common
import curve_interpolation
import implicits
import interpolation
import pickling
import plots


def constant_young(parameters, folder_path, data, data_interp, elem):
    print("* Plotting...")
    ncols = 3
    fig, axes = plots.plot_init(
        to_pgf=False,
        wide=False,
        ncols=ncols,
        nrows=2,
        aspectratio=0.8,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    for ax in axes[1]:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[0]:
        ax.set_xlabel(plots.labels["solid_fraction"])
    data_interp_span = data_interp["E_target_spans"][elem]
    label_target_young_val = r"$E_{t}=" + str(data_interp_span["young"]) + "$"
    contour_props = {}
    for i, prop in enumerate(["young_iso", "poisson_iso"]):
        contour_prop = scipy.interpolate.griddata(
            points=data_interp["interp_points"],
            values=data[prop],
            xi=data_interp_span["vertices"],
            method=interpolation.interp_method,
        )
        if i == 0:
            ax = axes[0][i]
            ax.set_ylabel(plots.labels[prop])
            plots.set_rasterized(
                ax.contourf(
                    data_interp["interp_βφ"]["solid_fraction"],
                    data_interp["interp_βφ"][prop],
                    data_interp["βG"],
                    cmap=None,
                    levels=0,
                    colors="silver",
                )
            )

            ax.plot(
                data_interp_span["solid_fraction"],
                contour_prop,
                "-.",
                lw=0.3,
                c="k",
                label=label_target_young_val,
            )
            ax.set_xlim([0, 1])
            if prop == "young_iso":
                ax.set_ylim([0, 1])
                axes[0][i].legend(loc="upper left")
        contour_props[prop] = contour_prop

    # plot interpolation curves
    poly_fit_funcs, tex_tau0_dict, tex_poly_dict = curve_interpolation.load_curve_interp(
        folder_path, target_young=data_interp_span["young"]
    )
    for i, prop in enumerate(["β", "φ"]):
        x, y = (
            data_interp_span["solid_fraction"],
            data_interp_span["vertices"][:, i],
        )
        ax = axes[0][1 + i]
        ax.plot(
            x,
            poly_fit_funcs[prop](x),
            lw=0.9,
            color="r",
            label=r"$" + plots.labels[prop + "_symb"] + r"_{E_{t}}(\tau)$",
        )
        ax.plot(x, y, linestyle="dashed", lw=0.6, c="k", label="Samples")
        ax.set(
            ylabel=plots.labels[prop + "_long"],
            ylim=[np.min(y), np.max(y)],
            xlim=[np.min(x), np.max(x)],
        )

        ax.legend(loc="upper right", fontsize=plots.fontsize - 2)

    solid_fractions_samples = np.linspace(*data_interp_span["solid_fraction_range"], ncols)
    for i, tau in enumerate(solid_fractions_samples):
        φ = poly_fit_funcs["φ"](tau)
        β = poly_fit_funcs["β"](tau)
        field = implicits.triangulation(
            n=parameters["n"],
            d=parameters["d"],
            φ=φ,
            β=β,
            resolution=parameters["resolution"],
            seed=parameters["seed"],
            num_points=parameters["num_points"],
            upsampling=parameters["upsampling"],
        )
        plots.plot_image_porous_structure(axes[1][i], field[: field.shape[0] // 2])
        axes[1][i].set_title(
            rf"$\tau \approx {tau:.2f} \quad \beta \approx {β:.2f}, \quad \varphi \approx {φ:.2f}$",
            y=-0.5,
            fontsize=plots.fontsize - 2,
        )

    plt.subplots_adjust(left=0.07, right=1, bottom=0, top=1, wspace=0.45, hspace=0.08)
    ext = curve_interpolation.target_young_percent(target_young=data_interp_span["young"])
    plots.save_pgf(folder_path, "constant_young_interp" + ext)

    # save some values for tex document
    with open(os.path.join(folder_path, "constantyoung" + ext + ".tex"), "w") as texfile:
        results_dict = {}
        common.write_value_tex_and_dict(
            texfile,
            "target_young" + ext,
            data_interp_span["young"],
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "target_young_tau_min" + ext,
            np.round(data_interp_span["solid_fraction_range"][0], 2),
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "target_young_tau_max" + ext,
            np.round(data_interp_span["solid_fraction_range"][1], 2),
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "target_young_poisson_min" + ext,
            np.floor(np.min(contour_props["poisson_iso"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "target_young_poisson_max" + ext,
            np.ceil(np.max(contour_props["poisson_iso"]) * 1000) / 1000.0,
            results_dict,
        )
        tau_min_target = (3 * data_interp_span["young"]) / (
            parameters["young_max"] + 2 * data_interp_span["young"]
        )
        print("* tau_min_target = " + str(tau_min_target))
        max_span_relative_coverage = data_interp_span["span_solid_fraction"] / (1 - tau_min_target)
        common.write_value_tex_and_dict(
            texfile,
            "max_span_relative_coverage_percent" + ext,
            np.round(max_span_relative_coverage * 100, 2),
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "tex_poly_tau" + ext,
            tex_tau0_dict["β"],
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "tex_poly_β" + ext,
            tex_poly_dict["β"],
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "tex_poly_φ" + ext,
            tex_poly_dict["φ"],
            results_dict,
        )


def main():
    parameters, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1])
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    with open(os.path.join(folder_path, "materialspace.tex"), "w") as texfile:
        results_dict = {}
        common.write_value_tex_and_dict(
            texfile,
            "poisson_min",
            np.floor(np.min(data["poisson_iso"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "poisson_max",
            np.ceil(np.max(data["poisson_iso"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "eff_young_min",
            np.floor(np.min(data["young_iso"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "eff_young_max",
            np.ceil(np.max(data["young_iso"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "solid_fraction_min",
            np.floor(np.min(data["solid_fraction"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "solid_fraction_max",
            np.ceil(np.max(data["solid_fraction"]) * 1000) / 1000.0,
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "maxdeviso",
            np.round(max(data["dev_iso"]), 3),
            results_dict,
        )
        common.write_value_tex_and_dict(
            texfile,
            "targetyounghalftonings",
            parameters["E_targets"][parameters["E_target_select_halftonings"]],
            results_dict,
        )
    for elem in range(len(data_interp["E_target_spans"])):
        constant_young(parameters, folder_path, data, data_interp, elem=elem)


if __name__ == "__main__":
    main()
