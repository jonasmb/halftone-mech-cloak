import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import common
import convergence
import plots


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1], prealias=convergence.alias)
    properties = [
        "bulk_iso",
        "shear_iso",
        "young_iso",
        "poisson_iso",
        "solid_fraction",
        "dev_iso",
    ]
    buckets = {prop: {} for prop in properties}
    for i, num_points in enumerate(data["num_points"]):
        for prop in properties:
            buckets[prop].setdefault(num_points, []).append(data[prop][i])

    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=1,
        nrows=6,
        aspectratio=0.15,
        subplot_opts={"sharex": True},
    )
    num_points_unique = np.unique(data["num_points"])
    for prop, ax in zip(properties, axes):
        min_val, max_val = np.min(data[prop]), np.max(data[prop])
        y_range = 0.1 * (max_val - min_val)
        if prop == "dev_iso":
            min_val = 0
            ax.set_ylim(0, max_val + y_range)
        else:
            ax.set_ylim(min_val - y_range, max_val + y_range)
        num_points_ini_exp = common.get_exp_power_two(int(num_points_unique[0]))
        num_points_end_exp = common.get_exp_power_two(int(num_points_unique[-1]))
        assert (num_points_ini_exp is not None) and (num_points_end_exp is not None)
        ax.set_xlim(2 ** (num_points_ini_exp - 0.1), 2 ** (num_points_end_exp + 0.1))
        ax.set_xscale("log", base=2)
        linewidths = plots.linewidth * 0.2
        ax.vlines(
            num_points_unique,
            ymin=min_val,
            ymax=max_val,
            linewidths=linewidths,
            colors="k",
            linestyles="dotted",
        )
        y_ticks = np.linspace(min_val, max_val, 4)
        ax.hlines(
            y_ticks,
            xmin=ax.get_xlim()[0],
            xmax=ax.get_xlim()[1],
            linewidths=linewidths,
            colors="k",
            linestyles="dotted",
        )
        # plot each mean
        mean_prop_per_num_points = [np.mean(buckets[prop][cat]) for cat in num_points_unique]
        ax.plot(
            num_points_unique,
            mean_prop_per_num_points,
            c="royalblue",
            marker=".",
            linestyle=":",
            markersize=plots.scatter_dot_size,
            linewidth=plots.linewidth,
            label="Mean",
        )

        mean_prop_per_num_points = [np.mean(buckets[prop][cat]) for cat in num_points_unique]
        ax.set_xticks(num_points_unique)
        ax.xaxis.set_tick_params(labelsize=plots.fontsize - 2)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_tick_params(labelsize=plots.fontsize - 2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
        ax.scatter(
            data["num_points"],
            data[prop],
            s=plots.scatter_dot_size,
            c="k",
            zorder=3,
            marker=".",
        )
        ax.set_ylabel(
            plots.labels[prop + "_short"],
            rotation=90,
            fontsize=plots.fontsize - 1,
        )
        ax.yaxis.label.set_horizontalalignment("right")
    axes[0].legend()
    axes[0].set_title(
        rf"$\beta={params['β'][convergence.convβid]}\quad \varphi={params['φ'][convergence.convφid]}$"
    )
    axes[-1].set_xlabel(plots.labels["num_points"])
    plt.subplots_adjust(left=0.15, right=1, bottom=0, top=1, wspace=0, hspace=0.15)
    plots.save_pgf(folder_path, "convergence", pad_inches=0.02)

    fig, axes = plots.plot_init(
        to_pgf=False,
        wide=False,
        ncols=len(num_points_unique),
        nrows=1,
        aspectratio=1,
    )
    for ax, num_points in zip(axes, num_points_unique):
        first_sample = data["field"][np.where(data["num_points"] == num_points)[0][0]]
        assert first_sample.shape[0] == first_sample.shape[1]
        resolution = first_sample.shape[0]
        label = (
            f"{plots.labels['num_points_short']} = $2^{{{common.get_exp_power_two(num_points)}}}$\n"
        )
        label += (
            f"{plots.labels['resolution_short']} = $2^{{{common.get_exp_power_two(resolution)}}}$"
        )
        ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
        fig.text(
            0.5,
            -0.39,
            label,
            ha="center",
            transform=ax.transAxes,
        )
        plots.plot_image_porous_structure(
            ax, data["field"][np.where(data["num_points"] == num_points)[0][0]]
        )

    plt.subplots_adjust(left=0, right=0.95, bottom=0, top=1, wspace=0.12, hspace=0)
    plots.save_pgf(folder_path, "convergence_gallery")


if __name__ == "__main__":
    main()
