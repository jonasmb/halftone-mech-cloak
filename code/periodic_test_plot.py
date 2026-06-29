import periodic_test
import common
import sys
import plots
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1], prealias=periodic_test.alias_periodic_test)
    num_points_unique = np.unique(data["num_points"])
    unique_target_young = np.unique(data["target_young"])
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=1,
        nrows=2 + len(unique_target_young) * 2,
        aspectratio=0.177,
        subplot_opts={"sharex": True},
    )

    def set_plot(ax, x_data, y_data, prop, y_range_off, target_young, n, min_val, max_val):
        num_points_ini_exp = common.get_exp_power_two(int(x_data[0]))
        num_points_end_exp = common.get_exp_power_two(int(x_data[-1]))
        ax.set_xlim(2 ** (num_points_ini_exp - 0.1), 2 ** (num_points_end_exp + 0.1))
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_data)
        ax.xaxis.set_tick_params(labelsize=plots.fontsize - 2)
        colors = plt.get_cmap("Dark2").colors
        target_young_line_thresh = 0.01
        y_range = y_range_off * (max_val - min_val)
        if prop == "dev_iso":
            min_val, max_val = 0, max_val + y_range
        elif prop == "young_iso":
            min_val = target_young - target_young_line_thresh * 0.2
            max_val = target_young + target_young_line_thresh * 1.3
        elif prop == "poisson_iso":
            min_val, max_val = min_val - y_range, max_val + y_range
        ax.set_ylim(min_val, max_val)
        ax.vlines(
            x_data,
            ymin=min_val,
            ymax=max_val,
            linewidths=plots.linewidth * 0.5,
            colors="k",
            linestyles="dotted",
        )
        if prop == "young_iso":
            ax.hlines(
                target_young,
                xmin=ax.get_xlim()[0],
                xmax=ax.get_xlim()[1],
                linewidths=plots.linewidth * 0.6,
                colors=colors[n],
                linestyles="solid",
            )
            ax.hlines(
                target_young + target_young_line_thresh,
                xmin=ax.get_xlim()[0],
                xmax=ax.get_xlim()[1],
                linewidths=plots.linewidth * 0.3,
                colors="black",
                linestyles="dashed",
            )
            ax.set_yticks([target_young, target_young + target_young_line_thresh])
        ax.set_ylabel(
            plots.labels[prop + "_short"],
            rotation=90,
            fontsize=plots.fontsize - 1,
        )
        ax.yaxis.label.set_horizontalalignment("right")
        label = plots.labels["target_young"] + " = " + str(target_young)
        ax.plot(
            x_data,
            y_data,
            c=colors[n],
            marker=".",
            linestyle=":",
            markersize=plots.scatter_dot_size * 0.6,
            linewidth=plots.linewidth * 0.5,
            label=label,
            zorder=3,
        )

    for k, prop in enumerate(["poisson_iso", "dev_iso", "young_iso"]):
        for i in range(len(unique_target_young)):
            num_points_data, prop_data, field_data = [], [], []
            for j in range(len(data["num_points"])):
                if data["target_young"][j] == unique_target_young[i]:
                    num_points_data.append(data["num_points"][j])
                    prop_data.append(data[prop][j])
                    field_data.append(data["field"][j])

            if prop == "young_iso":
                max_val, min_val = np.max(prop_data), np.min(prop_data)
                n = i * 2
                ax = axes[i * 2 + 1]
                ax.axis("off")
                spec = ax.get_subplotspec().subgridspec(1, len(field_data), wspace=0.1)
                ax.set_visible(False)
                for a, image in enumerate(field_data):
                    inner_ax = ax.figure.add_subplot(spec[0, a])
                    plots.noticks_equalaspect(inner_ax)
                    plots.plot_image_porous_structure(inner_ax, image)
            else:
                max_val, min_val = np.max(data[prop]), np.min(data[prop])
                if prop == "poisson_iso":
                    n = len(unique_target_young) * 2
                elif prop == "dev_iso":
                    n = len(unique_target_young) * 2 + 1
            set_plot(
                axes[n],
                num_points_data,
                prop_data,
                prop=prop,
                y_range_off=0.3,
                target_young=unique_target_young[i],
                n=i,
                max_val=max_val,
                min_val=min_val,
            )
            if prop == "young_iso":
                axes[i * 2].legend(loc="upper right", bbox_to_anchor=(1.05, 0.8))

    axes[-1].legend(loc="lower center", ncol=len(unique_target_young), bbox_to_anchor=(0.5, -0.8))
    axes[-1].set_xlabel(plots.labels["num_points"])
    plt.subplots_adjust(left=0.15, right=1, bottom=0, top=1, wspace=0, hspace=0.1)
    plots.save_pgf(folder_path, "periodic_test", pad_inches=0.01)


if __name__ == "__main__":
    main()
