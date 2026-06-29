import sys

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

import common
import fea_halftonings
import fea_validation
import pickling
import plots


def main():
    fig, axes_all = plots.plot_init(
        to_pgf=True,
        wide=True,
        ncols=6,
        nrows=1,
    )

    params, folder_path = common.parse_parameters()
    fea_data_dict = pickling.load_samples(
        sys.argv[1], prealias=fea_halftonings.alias_fea_halftonings
    )

    # plot schema
    axes = [axes_all[0], axes_all[2], axes_all[4]]
    inner_offset = 0.05
    external_color = "darkgray"
    internal_color = "lavender"
    floor_offset = 0.1
    arrow_offset = floor_offset * 0.7
    width_arrow = 0.01
    num_arrows = 4
    for c in np.linspace(0, 1, num_arrows):
        axes[0].arrow(x=c, y=1, dx=0, dy=arrow_offset, width=width_arrow, color="k")
        axes[0].arrow(x=c, y=0, dx=0, dy=-arrow_offset, width=width_arrow, color="k")
        axes[1].arrow(x=1, y=c, dx=arrow_offset, dy=0, width=width_arrow, color="k")
        axes[1].arrow(x=0, y=c, dx=-arrow_offset, dy=0, width=width_arrow, color="k")
    axes[0].set_title(plots.labels["vertical_load"])
    axes[1].set_title(plots.labels["horizontal_load"])
    axes[2].set_title(plots.labels["shear_load"])

    for c in [0.5, 1]:
        axes[2].arrow(
            x=c,
            y=-floor_offset * 0.5,
            dx=-0.35,
            dy=0,
            width=width_arrow,
            color="k",
        )
        axes[2].arrow(
            x=-floor_offset * 0.5,
            y=c,
            dx=0,
            dy=-0.35,
            width=width_arrow,
            color="k",
        )
        axes[2].arrow(
            x=1 - c,
            y=1 + floor_offset * 0.5,
            dx=+0.35,
            dy=0,
            width=width_arrow,
            color="k",
        )
        axes[2].arrow(
            x=1 + floor_offset * 0.5,
            y=1 - c,
            dx=0,
            dy=+0.35,
            width=width_arrow,
            color="k",
        )

    for ax in axes:
        linewidth = 0.2
        external_quad = matplotlib.patches.Rectangle(
            (0, 0),
            1,
            1,
            linewidth=linewidth,
            edgecolor="k",
            facecolor=external_color,
        )
        ax.add_patch(external_quad)
        internal_quad = matplotlib.patches.Rectangle(
            (inner_offset, inner_offset),
            1 - inner_offset * 2,
            1 - inner_offset * 2,
            linewidth=linewidth,
            edgecolor="k",
            facecolor=internal_color,
        )
        ax.add_patch(internal_quad)

        legend_handles = [
            matplotlib.patches.Rectangle(
                (0, 0), 1, 1, color=external_color, ec="black", lw=linewidth
            ),
            matplotlib.patches.Rectangle(
                (0, 0), 1, 1, color=internal_color, ec="black", lw=linewidth
            ),
        ]
        legend_labels = ["Solid external phase", "Porous structure"]
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, -0.2),
        )
        plots.noticks_equalaspect(ax)
        ax.axis("off")
        ax.set_xlim([-floor_offset * 2, 1 + floor_offset * 2])
        ax.set_ylim([-floor_offset * 2, 1 + floor_offset * 2])

    # plot deformed state (first result in database)
    axes = [axes_all[1], axes_all[3], axes_all[5]]
    for alias, fea_data_alias_dict in fea_data_dict["tests"].items():
        for (num_points, resolution), fea_data in fea_data_alias_dict.items():
            for mode_i, ax in enumerate(axes):
                print(ax)
                plots.noticks_equalaspect(ax)
                fea_validation.plot_fea(
                    ax,
                    fea_data["fea_results"][mode_i],
                    cmap_deformed_grid=plots.cmap_settings["von_mises_stress"],
                    dimnorm=resolution,
                    is_deformed_grid_plotted=False,
                )
            break
        break

    # plt.show()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0, hspace=0.3)
    plots.save_pgf(folder_path, "fea_boundary_conditions")


if __name__ == "__main__":
    main()
