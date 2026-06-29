import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy

import common
import implicits
import pixels_per_point
import plots


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1], prealias=pixels_per_point.alias)
    fig, (ax_mean, ax_var) = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=1,
        nrows=2,
        aspectratio=0.25,
        subplot_opts={"sharex": True},
    )
    nquad_epsabs = 2e-04
    nquad_limit = 1024
    line_style = "dashed"
    ppp_id_colors = np.array(["tab:blue", "tab:orange", "tab:green", "tab:purple"])
    data_ppp = np.divide(np.power(data["resolution"], 2), params["num_points"])
    data_var = np.array([np.var(image) for image in data["field"]])
    ax_mean.set_ylabel(plots.labels["expectation"])

    target_ppp = params["resolution"] ** 2 / params["num_points"]
    print("* Target ppp = " + str(target_ppp))

    def scatter_plot(ax, y):
        ax.scatter(
            data_ppp,
            y,
            s=plots.scatter_dot_size,
            c=ppp_id_colors[data["ppp_id"]],
            linewidths=plots.linewidthscatter,
            edgecolors="k",
            marker="o",
            zorder=3,
        )
        ax.set_xscale("log", base=2)
        ly = np.max(y) - np.min(y)
        lyf = 0.15
        ylim = [np.min(y) - ly * lyf, np.max(y) + ly * lyf]
        ax.vlines(
            x=target_ppp,
            ymin=ylim[0],
            ymax=ylim[1],
            linewidth=plots.linewidth * 0.5,
            linestyles="dotted",
            colors="gray",
            zorder=0,
        )
        ax.set_ylim(ylim)

    scatter_plot(ax_mean, data["solid_fraction"])

    ax_var.set_ylabel(plots.labels["variance"])
    ax_var.set_xlabel(plots.labels["pixels_per_point"])
    scatter_plot(ax_var, data_var)
    xlim = [np.min(data_ppp) * 0.5, np.max(data_ppp) + np.min(data_ppp) * 0.5]
    ax_var.set_xlim(xlim)
    ax_var.set_xticks(np.unique(data_ppp))

    # numerical integration
    mean_max_percent_error_all = []
    variance_max_percent_error_all = []
    ppp_id = 0
    for β in params["β"]:
        for φ in params["φ"]:
            print(f"* β = {β}   φ = {φ}")

            def Sλ_params(λ1, λ2):
                return implicits.Sλ(
                    λ1=λ1,
                    λ2=λ2,
                    α=0,
                    n=params["n"],
                    d=params["d"],
                    φ=φ,
                    β=β,
                )

            λ1_limits = [0, 1]

            def λ2_limits(λ1):
                return [0, 1 - λ1]

            opts = {"limit": nquad_limit, "epsabs": nquad_epsabs, "epsrel": 0}
            result, error = scipy.integrate.nquad(
                Sλ_params, [λ2_limits, λ1_limits], opts=[opts, opts]
            )
            print("\t* Error given by scipy.integrate.nquad = " + str(error))
            mean = 2 * result
            print("\t* Expectation = " + str(mean))
            variance = mean * (1 - mean)
            print("\t* Variance = " + str(variance))
            target_ppp_ids = np.where(
                np.logical_and(data_ppp == target_ppp, np.array(data["ppp_id"]) == ppp_id)
            )
            if target_ppp_ids[0].size == 0:
                print("\t! WARNING: target_ppp_ids[0].size == 0")
                continue

            def percent_error_upper_bound(v, v_approx):
                return np.max(np.ceil(100 * 100 * np.abs(np.divide(v - v_approx, v))) / 100.0)

            mean_max_percent_error = percent_error_upper_bound(
                mean,
                np.array(data["solid_fraction"])[target_ppp_ids],
            )
            mean_max_percent_error_all.append(mean_max_percent_error)
            print("\t\t- Target ppp: Mean percent error <= " + str(mean_max_percent_error) + "%")
            variance_max_percent_error = percent_error_upper_bound(
                variance,
                data_var[target_ppp_ids],
            )
            variance_max_percent_error_all.append(variance_max_percent_error)
            print(
                "\t\t- Target ppp: Variance percent error <= "
                + str(variance_max_percent_error)
                + "%"
            )
            ax_mean.hlines(
                y=mean,
                xmin=xlim[0],
                xmax=xlim[1],
                linewidth=plots.linewidth,
                linestyles=line_style,
                colors=ppp_id_colors[ppp_id],
            )
            ax_var.hlines(
                y=variance,
                xmin=xlim[0],
                xmax=xlim[1],
                linewidth=plots.linewidth,
                linestyles=line_style,
                colors=ppp_id_colors[ppp_id],
                label=r"$\beta=" + str(β) + r"\quad\varphi=" + str(φ) + "$",
            )
            ppp_id += 1
    ax_var.legend(
        loc="lower center",
        columnspacing=1.0,
        labelspacing=0.25,
        ncols=2,
        bbox_to_anchor=(0.5, -0.55),
        fontsize=plots.fontsize - 1,
    )
    if len(mean_max_percent_error_all) > 0:
        max_error_mean = np.max(mean_max_percent_error_all)
        max_error_variance = np.max(variance_max_percent_error_all)
        print("* Target ppp: Max error mean = " + str(max_error_mean) + "%")
        print("* Target ppp: Max error variance = " + str(max_error_variance) + "%")
    else:
        print("!WARNING: len(mean_max_percent_error_all) == 0")
        max_error_mean = max_error_variance = "\\infty"
    with open(
        os.path.join(folder_path, "convergence_pixels_per_point.tex"),
        "w",
    ) as texfile:
        common.write_value_tex(texfile, "ppp_max_error_mean", max_error_mean)
        common.write_value_tex(texfile, "ppp_max_error_variance", max_error_variance)

    plt.subplots_adjust(left=0.1, right=1, bottom=0, top=1, wspace=0, hspace=0.1)
    plots.save_pgf(folder_path, "pixels_per_point")


if __name__ == "__main__":
    main()
