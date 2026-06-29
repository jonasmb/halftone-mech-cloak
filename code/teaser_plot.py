import os

import matplotlib.pyplot as plt
import numpy as np

import common
import plots
import teaser


def plot_render(ax, title, name, folder_path, cmap=None):
    ax.set_title(title)
    image = plt.imread(os.path.join(folder_path, name))
    ax.imshow(image, interpolation=plots.imshowinterp, aspect="equal", origin="upper")


def main():
    params, folder_path = common.parse_parameters()
    fig, axes = plots.plot_init(to_pgf=True, wide=True, ncols=5, nrows=1, aspectratio=1)
    for ax in axes:
        plots.noticks_equalaspect(ax)
    (
        ax_target,
        ax_result,
        ax_render_top,
        ax_render_close1,
        ax_render_close2,
    ) = axes
    # plot precomputed renders of thin plate and teaser
    ax_target.set_title(plots.labels["target_solid_fraction"])
    ax_result.set_title("Result")
    image_result = plt.imread(os.path.join(folder_path, "teaser.png"))
    ax_result.imshow(
        image_result,
        interpolation=plots.imshowinterp,
        aspect="equal",
        origin="lower",
        cmap=plots.cmap_settings["mat"],
    )
    plot_render(ax_render_top, "Orthographic render", "render_top_view.png", folder_path)
    plot_render(ax_render_close1, "First close-up render", "render_closeup_2.png", folder_path)
    plot_render(ax_render_close2, "Second close-up render", "render_closeup_5.png", folder_path)
    func_teaser = teaser.get_field_teaser(params, folder_path, compute_field=False)
    X = np.meshgrid(np.linspace(0, 1, plots.subdiv_val), np.linspace(0, 1, plots.subdiv_val))
    plots.set_rasterized(
        ax_target.contourf(
            X[0],
            X[1],
            func_teaser(X),
            vmin=0,
            vmax=1,
            levels=plots.subdiv_val,
            cmap=plots.cmap_settings["mat"],
        )
    )
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.03, hspace=0)
    print("* Saving plot...")
    plots.save_pgf(folder_path, "teaser")


if __name__ == "__main__":
    main()
