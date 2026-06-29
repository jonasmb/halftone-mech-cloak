import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import common
import mean_fluctuations
import plots
import varying_poisson


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1], prealias=varying_poisson.alias_varying_poisson)
    props = ("young_iso", "poisson_iso")

    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=len(props),
        nrows=1,
        aspectratio=0.8,
    )
    data["solid_fraction"] = np.array(data["solid_fraction"])
    data["poisson_iso"] = np.array(data["poisson_iso"])
    data["young_iso"] = np.array(data["young_iso"])
    for v in np.unique(data["poisson_ratio"]):
        idx = np.where(data["poisson_ratio"] == v)[0]
        solid_fractions = np.concatenate((data["solid_fraction"][idx], [1]), axis=0)
        poisson_ratios = np.concatenate((data["poisson_iso"][idx], [v]), axis=0)
        axes[1].plot(
            solid_fractions,
            poisson_ratios,
            c="k",
            lw=plots.linewidth * 0.1,
            zorder=0,
            linestyle="dotted",
        )

    solid_frac_list = [0]
    mean_young_list = [0]
    coeff_var_max = 0
    for β in np.unique(data["β"]):
        idx = np.where(data["β"] == β)[0]
        solid_frac_list.append(np.mean(data["solid_fraction"][idx]))
        mean_young_list.append(np.mean(data["young_iso"][idx]))
        mean, std, coeff_var = mean_fluctuations.get_stats(data["young_iso"][idx])
        coeff_var_max = max(coeff_var_max, coeff_var)
    print("* Max CV of E among all different β =" + str(coeff_var_max))
    with open(os.path.join(folder_path, "varying_poisson.tex"), "w") as texfile:
        common.write_value_tex(
            texfile,
            "varpoissoncoeffvarmaxyoung",
            common.decimal_to_latex_mantissa(coeff_var_max, rnd=3),
        )

    solid_frac_list.append(1)
    mean_young_list.append(1)
    axes[0].plot(
        solid_frac_list,
        mean_young_list,
        c="k",
        lw=plots.linewidth * 0.1,
        zorder=0,
        linestyle="dotted",
    )

    xfield, colorfield = "solid_fraction", "poisson_ratio"
    for ax, prop in zip(axes, props):
        ax.set_xlabel(plots.labels[xfield])
        ax.set_ylabel(plots.labels[prop], labelpad=-0.2)
        ax.set_xlim([0, 1])
        cm = ax.scatter(
            data[xfield],
            data[prop],
            s=plots.scatter_dot_size * 0.25,
            c=data[colorfield],
            cmap=plots.cmap_settings[colorfield],
            marker=".",
        )
        ax.locator_params(axis="both", nbins=plots.nbins)
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim(params["varying_poisson"])
    axes[1].hlines(
        0,
        xmin=0,
        xmax=1,
        linewidths=plots.linewidth * 0.4,
        colors="k",
        linestyles="solid",
    )
    fig.colorbar(
        cm,
        ax=axes[1],
        label=plots.labels[colorfield],
        ticks=np.linspace(
            np.min(data[colorfield]),
            np.max(data[colorfield]),
            plots.global_num_ticks_cb,
        ),
    )
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0, top=1, wspace=0.3, hspace=0.0)
    plots.save_pgf(folder_path, "varying_poisson", pad_inches=0.02)


if __name__ == "__main__":
    main()
