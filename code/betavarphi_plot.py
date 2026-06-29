import sys

import matplotlib.pyplot as plt

import common
import interpolation
import pickling
import plots


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1])
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    fig, axes = plots.plot_init(
        to_pgf=False,
        wide=False,
        ncols=2,
        nrows=len(data_interp["interp_βφ"]),
        aspectratio=0.7,
        subplot_opts={"sharex": True, "sharey": True},
    )
    for r, (prop, ax_row) in enumerate(zip(data_interp["interp_βφ"], axes)):
        for c, ax in enumerate(ax_row):
            ax.set(
                **{
                    "xlim": params["β"],
                    "ylim": params["φ"],
                    "xlabel": r"$\beta$" if r == len(axes) - 1 else "",
                    "ylabel": r"$\varphi$" if c == 0 else "",
                }
            )
            ax.yaxis.label.set_rotation(0)

        plots.scatterp(
            fig,
            ax_row[0],
            ax_row[1],
            data["β"],
            data["φ"],
            data_interp["βG"],
            data_interp["φG"],
            c=data[prop],
            c_interp=data_interp["interp_βφ"][prop],
            cmap=plots.cmap_settings[prop],
            label=plots.labels[prop],
        )
    plt.subplots_adjust(left=0.1, right=1, bottom=0, top=1, wspace=0.15, hspace=0.15)
    plots.save_pgf(folder_path, "betavarphi")


if __name__ == "__main__":
    main()
