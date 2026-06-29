import sys

import matplotlib.pyplot as plt
import numpy as np

import common
import fea_halftonings
import fea_validation
import pickling
import plots


def main():
    params, folder_path = common.parse_parameters()
    fea_data_dict = pickling.load_samples(
        sys.argv[1], prealias=fea_halftonings.alias_fea_halftonings
    )

    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=True,
        ncols=6,
        nrows=len(fea_data_dict["tests"].items()),
        subplot_opts={"sharey": True},
    )
    i = 0
    for alias, fea_data_alias_dict in fea_data_dict["tests"].items():
        print("* " + alias)
        j = 0
        for (num_points, resolution), fea_data in fea_data_alias_dict.items():
            if not (num_points == 64 or num_points == 1024):
                continue
            ax = None
            print((num_points, resolution))
            for mode_i in range(3):
                if len(fea_data_dict["tests"]) == 1:
                    ax = axes[j * 3 + mode_i]
                elif len(fea_data_dict["tests"]) > 1:
                    ax = axes[i][j * 3 + mode_i]
                assert ax is not None

                plots.noticks_equalaspect(ax)
                if i == 0 and mode_i == 1:
                    ax.set_title(
                        plots.labels["num_points_short"]
                        + "="
                        + str(num_points)
                        + " "
                        + plots.labels["resolution_short"]
                        + "="
                        + str(resolution)
                    )
                if i == len(fea_data_dict["tests"]) - 1:
                    label = None
                    mode_key = fea_data["fea_results"][mode_i]["mode"]
                    if mode_key == "x-tension":
                        label = plots.labels["vertical_load_short"]
                    elif mode_key == "y-tension":
                        label = plots.labels["horizontal_load_short"]
                    elif mode_key == "xy-shear":
                        label = plots.labels["shear_load"]
                    print(label)
                    ax.text(
                        0.2,
                        -0.1,
                        label,
                        ha="left",
                        va="center",
                        fontsize=plots.fontsize - 1,
                    )
                if fea_data is not None:
                    pcmesh = fea_validation.plot_fea(
                        ax,
                        fea_data["fea_results"][mode_i],
                        cmap_deformed_grid=plots.cmap_settings["von_mises_stress"],
                        dimnorm=resolution,
                    )

            j += 1
        i += 1
    plt.subplots_adjust(left=0.0, right=1, bottom=0, top=1, wspace=0.03, hspace=0.12)
    # representative colorbar
    fig.colorbar(
        pcmesh,
        ax=axes,
        orientation="horizontal",
        cmap=plots.cmap_settings["von_mises_stress"],
        label="von Mises stress",
        shrink=0.2,
        fraction=0.1,
        pad=0.05,
        ticks=[],
    )
    plots.save_pgf(folder_path, fea_halftonings.alias_fea_halftonings)


if __name__ == "__main__":
    main()
