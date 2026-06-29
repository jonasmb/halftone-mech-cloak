import sys

import matplotlib.pyplot as plt
import numpy as np

import common
import mean_fluctuations
import pickling
import plots


def main():
    params, folder_path = common.parse_parameters()
    mean_fluc_number_plot_samples = 5
    results = pickling.load_samples(sys.argv[1], prealias=mean_fluctuations.alias)
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=mean_fluc_number_plot_samples + 1,
        nrows=2,
        aspectratio=0.8,
    )
    i_list = list(range(mean_fluc_number_plot_samples - 1)) + [params["mean_fluc_num_samples"] - 1]

    def plot_sample_th(ax, field, title):
        plots.noticks_equalaspect(ax)
        plots.plot_image_porous_structure(ax, field)
        if title:
            title_i = "$"
            if i == params["mean_fluc_num_samples"] - 1:
                title_i += r"\ldots \quad"
            title_i += str(i + 1) + "$"
            ax.set_title(title_i, fontsize=plots.fontsize - 1)

    k = 0
    print(axes.shape)
    for i in i_list:
        plot_sample_th(axes[0][k], results["tri_thick_fields"][i], title=True)
        plot_sample_th(axes[1][k], results["S_fields"][i], title=False)
        k += 1

    def plot_stats(ax, X, color_CV):
        ax.axis("off")
        mu, sigma, CV = mean_fluctuations.get_stats(X)
        rnd = 3
        ax.text(
            0,
            0.9,
            r"$\mu = " + str(np.round(mu, rnd)) + "$",
            fontsize=plots.fontsize - 2,
        )

        ax.text(
            0,
            0.6,
            r"$\sigma = " + common.decimal_to_latex_mantissa(sigma, rnd) + "$",
            fontsize=plots.fontsize - 2,
        )
        ax.text(
            0,
            0.3,
            r"$CV = " + common.decimal_to_latex_mantissa(CV, rnd) + "$",
            fontsize=plots.fontsize - 2,
            color=color_CV,
            fontweight=90,
        )

    plot_stats(axes[0][-1], results["tri_thick_means"], color_CV="darkred")
    plot_stats(axes[1][-1], results["S_means"], color_CV="darkblue")

    # plt.show()
    plt.subplots_adjust(left=0.0, right=0.95, bottom=0, top=1, wspace=0.0, hspace=0.09)
    plots.save_pgf(folder_path, "mean_fluctuations")
    return


if __name__ == "__main__":
    main()
