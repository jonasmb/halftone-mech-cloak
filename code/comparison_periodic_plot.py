import matplotlib.pyplot as plt
import numpy as np

import common
import implicits
import plots


def power_spectrum(I):
    F = np.fft.fft2(I)  # fourier transform
    F[0][0] = 0  # set DC component to zero
    return np.abs(np.fft.fftshift(F)) ** 2


def plot_field(ax, field, cmap, title=None):
    ax.set_title(title)
    ax.imshow(
        field,
        interpolation=plots.imshowinterp,
        aspect="equal",
        cmap=cmap,
    )


def main():
    # parameters
    resolution = 4096
    upsampling = 4
    # two lattice points per square (below repeated for periodicity)
    # perturb center point to avoid coincident case Delaunay triangulation
    lattice_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5 + 1e-10, 0.5]])
    tiling_factor = 16
    num_points = 2 * tiling_factor**2

    params, folder_path = common.parse_parameters()
    φ = (params["φ"][0] + params["φ"][1]) * 0.5
    β = (params["β"][0] + params["β"][1]) * 0.5
    fig, axes = plots.plot_init(to_pgf=True, wide=False, ncols=5, nrows=1)
    for ax in axes.flatten():
        plots.noticks_equalaspect(ax)
    print("* Resolution = " + str(resolution))
    field_periodic = implicits.triangulation(
        n=params["n"],
        d=params["d"],
        φ=φ,
        β=β,
        resolution=resolution,
        seed=params["seed"],
        num_points=None,
        upsampling=upsampling,
        input_points=lattice_points,
    )

    field_ours = implicits.triangulation(
        n=params["n"],
        d=params["d"],
        φ=φ,
        β=β,
        resolution=resolution,
        seed=params["seed"],
        num_points=num_points,
        upsampling=upsampling,
    )

    print("* Computing FFT measures...")
    # plot spectral measures
    field_periodic_spec = power_spectrum(field_periodic)
    field_ours_spec = power_spectrum(field_ours)

    # modify to logarithmic scale for better visualization
    field_periodic_spec_log = np.log1p(1 + field_periodic_spec)
    field_ours_spec_log = np.log1p(1 + field_ours_spec)

    plot_field(
        axes[0],
        field_periodic,
        plots.cmap_settings["mat"],
        r"$\mathcal{Z}_{lat}$",
    )
    plot_field(
        axes[1],
        np.tile(field_periodic, (tiling_factor, tiling_factor)),
        plots.cmap_settings["mat"],
        r"$\mathcal{Z}_{lat} " + str(tiling_factor) + "\\times" + str(tiling_factor) + "$",
    )
    plot_field(
        axes[2],
        field_periodic_spec_log,
        plots.cmap_settings["spectral"],
        r"$P(\mathcal{Z}_{lat})$",
    )
    plot_field(axes[3], field_ours, plots.cmap_settings["mat"], r"$\mathcal{Z}_{our}$")
    plot_field(
        axes[4],
        field_ours_spec_log,
        plots.cmap_settings["spectral"],
        r"$P(\mathcal{Z}_{our})$",
    )

    # plt.show()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.06, hspace=0.34)
    plots.save_pgf(folder_path, "comparison_periodic")


if __name__ == "__main__":
    main()
