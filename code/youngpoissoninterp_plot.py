import sys

import matplotlib.pyplot as plt
import numpy as np

import common
import interpolation
import pickling
import plots


def plot_poisson_interp(parameters, data, data_interp, folder_path):
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=2,
        nrows=1,
        aspectratio=0.7,
    )

    plots.plot_data(
        ax_data=[data[k] if k in data else None for k in ["poisson_iso"] * 2],
        ax_all=axes.flatten(),
        ax_color=[data[k] if k in data else None for k in ["β", "φ"]],
        ax_cmap=[plots.cmap_settings[k] for k in ["β", "φ"]],
        ax_ylabel=[plots.labels.get(k, None) for k in ["poisson_iso", None]],
        ax_top=[np.max(data["poisson_iso"]) + plots.poisson_y_off] * 2,
        ax_bottom=[np.min(data["poisson_iso"]) - plots.poisson_y_off] * 2,
        ax_cb=[r"$\beta$", r"$\varphi$"],
        solid_fraction=data["solid_fraction"],
        x_label_mask=[True] * 2,
        y_ticks_mask=[True, False],
        x_ticks_mask=[True] * 2,
    )
    plt.subplots_adjust(left=0.05, right=1, bottom=0, top=1, wspace=0.15, hspace=0.1)
    plots.save_pgf(folder_path, "poisson_interp")


def plot_young_interp(parameters, data, data_interp, folder_path):
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=True,
        ncols=4,
        nrows=1,
        aspectratio=0.7,
    )
    plots.plot_data(
        ax_data=[data[k] if k in data else None for k in ["young_iso", None] * 2],
        ax_all=axes.flatten(),
        ax_color=[data[k] if k in data else None for k in ["β", None, "φ", None]],
        ax_cmap=[plots.cmap_settings[k] for k in ["β", "β", "φ", "φ"]],
        ax_ylabel=[plots.labels.get(k, None) for k in ["young_iso", None, None, None]],
        ax_top=[parameters["young_max"]] * 4,
        ax_bottom=[0] * 4,
        ax_cb=[r"$\beta$", None, r"$\varphi$", None],
        solid_fraction=data["solid_fraction"],
        x_label_mask=[True] * 4,
        y_ticks_mask=[True, False, False, False],
        x_ticks_mask=[True] * 4,
    )
    prop = "young_iso"
    for j, param in enumerate(["β", "φ"]):
        ax = axes[2 * j + 1]
        plots.set_rasterized(
            ax.contourf(
                data_interp["interp_βφ"]["solid_fraction"],
                data_interp["interp_βφ"][prop],
                data_interp[f"{param}G"],
                cmap=plots.cmap_settings[param],
                levels=256,
            ),
        )
        # print wireframe for better understanding
        for a in range(parameters["num_samples_param"]):
            x1, y1 = [], []
            x2, y2 = [], []
            for b in range(parameters["num_samples_param"]):
                ab1 = a * parameters["num_samples_param"] + b
                ab2 = b * parameters["num_samples_param"] + a
                x1.append(data["solid_fraction"][ab1])
                y1.append(data[prop][ab1])
                x2.append(data["solid_fraction"][ab2])
                y2.append(data[prop][ab2])
            ax.plot(x1, y1, "-", c="darkgray", linewidth=0.001)
            ax.plot(x2, y2, "-", c="darkgray", linewidth=0.001)

    for k in range(4):
        common.plot_hs_upper_bound(ax=axes[k], name=prop, params=parameters)
    plt.subplots_adjust(left=0.05, right=1, bottom=0, top=1, wspace=0.15, hspace=0.1)
    plots.save_pgf(folder_path, "young_interp", pad_inches=0.1)


def main():
    parameters, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1])
    data_interp = pickling.load_samples(
        sys.argv[1],
        prealias=interpolation.alias_interpolation,
    )

    plot_poisson_interp(parameters, data, data_interp, folder_path)
    plot_young_interp(parameters, data, data_interp, folder_path)


if __name__ == "__main__":
    main()
