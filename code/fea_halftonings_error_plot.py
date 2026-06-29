import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import common
import fea_halftonings
import fea_validation
import pickling
import plots
import cycler


def main():
    params, folder_path = common.parse_parameters()
    fea_data_dict = pickling.load_samples(
        sys.argv[1], prealias=fea_halftonings.alias_fea_halftonings
    )
    fig, axes = plots.plot_init(
        to_pgf=True, wide=True, ncols=3, nrows=1, aspectratio=0.5, subplot_opts={"sharey": True}
    )
    axes_dict = {"x-tension": axes[0], "y-tension": axes[1], "xy-shear": axes[2]}
    for alias, fea_data_alias_dict in fea_data_dict["tests"].items():
        plot_data = {"x-tension": [[], []], "y-tension": [[], []], "xy-shear": [[], []]}
        for (num_points, resolution), fea_data in fea_data_alias_dict.items():
            fea_results = fea_data["fea_results"]
            for fea_result in fea_results:
                plot_data[fea_result["mode"]][0].append((num_points, resolution))
                plot_data[fea_result["mode"]][1].append(fea_result["disp_error"])
        print(plot_data)
        for mode in axes_dict:
            x_data, y_data = plot_data[mode][0], plot_data[mode][1]
            x_pos = range(len(x_data))
            x_data_num_points = []
            for i in range(len(x_data)):
                x_data_num_points.append(x_data[i][0])
            axes_dict[mode].set_xticks(x_pos)
            axes_dict[mode].set_xticklabels(x_data_num_points)
            axes_dict[mode].semilogy(
                x_pos,
                y_data,
                label=plots.labels[alias],
                linewidth=0.2,
                marker="o",
                markersize=0.8,
            )
    for mode, ax in axes_dict.items():
        ax.set_xlabel(plots.labels["num_points_short"])
        if mode == "x-tension":
            ax.set_ylabel(plots.labels["disp_error"] + "(log scale)")
        ax.set_prop_cycle(cycler.cycler(color=plots.tol_muted))
        ax.autoscale_view(scaley=True)
        if mode == "x-tension":
            ax.set_title(plots.labels["vertical_load"])
        elif mode == "y-tension":
            ax.set_title(plots.labels["horizontal_load"])
        elif mode == "xy-shear":
            ax.set_title(plots.labels["shear_load"])
        ax.grid(True, which="major", axis="x", linewidth=0.15)
        ax.grid(True, which="major", axis="y", linewidth=0.15)
        ax.grid(True, which="minor", axis="y", linewidth=0.1, linestyle=":")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(labels),
        frameon=False,
    )
    ax0 = axes[0]
    ymin, ymax = ax0.get_ylim()
    # round to nearest powers of 10
    ymin = 10 ** np.floor(np.log10(ymin))
    ymax = 10 ** np.ceil(np.log10(ymax))
    ax0.set_ylim(ymin, ymax)
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0, top=1, wspace=0.07, hspace=0.0)
    plots.save_pgf(folder_path, fea_halftonings.alias_fea_halftonings + "_error_plot")


if __name__ == "__main__":
    main()
