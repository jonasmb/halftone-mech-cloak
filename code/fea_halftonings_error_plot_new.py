import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

import common
import fea_halftonings
import fea_validation
import pickling
import plots
import cycler

def compute_global_displacement_error(data_dict):
    U = data_dict.get("nodal_displacements")
    U_t = data_dict.get("homogeneous_nodal_displacements")
    return np.linalg.norm(U - U_t) / np.linalg.norm(U_t)

def plot_global_error(params, folder_path, fea_data_dict):
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=True,
        ncols=3,
        nrows=1,
        aspectratio=1.5,
        subplot_opts={"sharey": True}
    )
    axes_dict = {"x-tension": axes[0], "y-tension": axes[1], "xy-shear": axes[2]}
    for alias, fea_data_alias_dict in fea_data_dict["tests"].items():
        plot_data = {"x-tension":[[], []], "y-tension":[[], []], "xy-shear":[[], []]}
        for (num_points, resolution), fea_data in fea_data_alias_dict.items():
            fea_results = fea_data["fea_results"]
            for fea_result in fea_results:
                plot_data[fea_result["mode"]][0].append((num_points, resolution))
                disp_error = compute_global_displacement_error(fea_result)
                plot_data[fea_result["mode"]][1].append(disp_error)
        print(plot_data)
        for mode in axes_dict:
            x_data, y_data = plot_data[mode][0], plot_data[mode][1]
            x_pos = range(len(x_data))
            x_data_num_points = []
            for i in range(len(x_data)):
                x_data_num_points.append(x_data[i][0])
            axes_dict[mode].set_xticks(x_pos)
            axes_dict[mode].set_xticklabels(x_data_num_points)
            axes_dict[mode].semilogy(
                x_pos,
                y_data,
                label=plots.labels[alias],
                linewidth=0.2,
                marker="o",
                markersize=0.8,
            )
    for (mode, ax) in axes_dict.items():
        ax.set_xlabel(plots.labels["num_points_short"])
        if mode == "x-tension":
            ax.set_ylabel(plots.labels["disp_error"] + "(log scale)")
        ax.set_prop_cycle(cycler.cycler(color=plots.tol_muted))
        ax.autoscale_view(scaley=True)
        if mode == "x-tension":
            ax.set_title(plots.labels["vertical_load"])
        elif mode == "y-tension":
            ax.set_title(plots.labels["horizontal_load"])
        elif mode == "xy-shear":
            ax.set_title(plots.labels["shear_load"])
        ax.grid(True, which='major', axis='x', linewidth=0.15)
        ax.grid(True, which='major', axis='y', linewidth=0.15)
        ax.grid(True, which='minor', axis='y', linewidth=0.1, linestyle=':')
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15), 
        ncol=len(labels),
        frameon=False
    )   
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0, top=1, wspace=0.07, hspace=0.0)
    #plots.save_pgf(folder_path, fea_halftonings.alias_fea_halftonings + "_error_plot")


def plot_displacement_error(ax, data_dict, nel, cmap, vmin=None, vmax=None):
    U = data_dict.get("nodal_displacements")
    U_t = data_dict.get("homogeneous_nodal_displacements")
    grid = fea_validation.Grid(nel, nel, nel, nel)
    dx = U[0::2] - U_t[0::2]
    dy = U[1::2] - U_t[1::2]
    error_mag = (dx*dx + dy*dy) / np.sum(U_t * U_t)
    
    # sanity check
    rel_error = np.linalg.norm(U - U_t) / np.linalg.norm(U_t)
    assert np.isclose(np.sum(error_mag), rel_error**2)

    X = np.reshape(grid.X, (grid.nny, grid.nnx), order="F")
    Y = np.reshape(grid.Y, (grid.nny, grid.nnx), order="F")
    C = np.reshape(error_mag, (grid.nny, grid.nnx), order="F")

    relative_modulus = data.get("relative_modulus")
    alpha = np.ceil(relative_modulus).flatten()
    
    pcm = ax.pcolormesh(X, Y, C, alpha=alpha, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    plots.noticks_equalaspect(ax)
    return pcm, error_mag

def plot_matched_histogram(ax, data, cmap_name):
    ax.hist(data, bins=128, density=True, rasterized=True)

def plot_local_error(params, folder_path, fea_data_dict):
    cmap = "turbo"
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=True,
        ncols=len(fea_data_dict["tests"]),
        nrows=2 * 3,
    )
    i = 0
    for alias, fea_data_alias_dict in fea_data_dict["tests"].items():
        axes[0][i].set_title(plots.labels[alias])
        (num_points, resolution), fea_data = list(fea_data_alias_dict.items())[-1]
        for j, fea_result in enumerate(fea_data["fea_results"]):
            print(fea_result["mode"])
            pcm, err_mag = plot_displacement_error(axes[j * 2][i], fea_result, resolution, cmap)
            plot_matched_histogram(axes[j * 2 + 1][i], err_mag.flatten(), cmap)
        i += 1
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0, top=1, wspace=0.2, hspace=0.1)
    plots.save_pgf(folder_path, fea_halftonings.alias_fea_halftonings + "_error_plot_local")

def main():
    params, folder_path = common.parse_parameters()
    fea_data_dict = pickling.load_samples(
        sys.argv[1],
        #prealias=fea_halftonings.alias_fea_halftonings
        prealias="fea_halftonings_small_test",
    )
    # plot_global_error(params, folder_path, fea_data_dict)
    plot_local_error(params, folder_path, fea_data_dict)

    
if __name__ == "__main__":
    main()
