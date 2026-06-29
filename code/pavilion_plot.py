import matplotlib.pyplot as plt

import common
import plots
import teaser_plot


def main():
    params, folder_path = common.parse_parameters()
    fig, axes = plots.plot_init(to_pgf=True, wide=True, ncols=2, nrows=2, aspectratio=1)
    for ax in axes.flatten():
        plots.noticks_equalaspect(ax)
    (ax_front, ax_back), (ax_closeup, ax_closeup_second) = axes[0], axes[1]

    # plot precomputed renders
    teaser_plot.plot_render(ax_front, "Front render", "pavilion_front.png", folder_path)
    teaser_plot.plot_render(ax_back, "Back render", "pavilion_back.png", folder_path)
    teaser_plot.plot_render(
        ax_closeup, "First close-up render", "pavilion_closeup.png", folder_path
    )
    teaser_plot.plot_render(
        ax_closeup_second, "Second close-up render", "pavilion_closeup_second.png", folder_path
    )
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.03, hspace=-0.55)
    print("* Saving plot...")
    plots.save_pgf(folder_path, "pavilion")


if __name__ == "__main__":
    main()
