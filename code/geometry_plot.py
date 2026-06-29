import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import common
import implicits
import plots

plot_overview_geometry_triangulation_resolution = 2048


def plot_overview_geometry(parameters, folder_path):
    fig, (ax_points, ax_triangles, ax_pores) = plots.plot_init(
        to_pgf=True, wide=False, ncols=3, nrows=1
    )

    def default_overview(ax):
        plots.noticks_equalaspect(ax)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    factor_φ = 0.5
    factor_β = 0.5
    field = implicits.triangulation(
        n=parameters["n"],
        d=parameters["d"],
        φ=(parameters["φ"][0] + parameters["φ"][1]) * factor_φ,
        β=(parameters["β"][0] + parameters["β"][1]) * factor_β,
        resolution=plot_overview_geometry_triangulation_resolution,
        seed=parameters["seed"],
        num_points=parameters["num_points"],
        upsampling=1,
    )
    ax_pores.imshow(
        field,
        vmin=0,
        vmax=1,
        extent=[0, 1, 0, 1],
        interpolation=plots.imshowinterp,
        cmap=plots.cmap_settings["mat"],
        zorder=0,
    )
    points = np.array(
        implicits.ccvt_wrapper(parameters["num_points"], random_seed=parameters["seed"])
    )

    lw_base = 0.05
    point_size = 0.4
    color_points = "dodgerblue"
    color_edges = "peru"

    def plot_points(ax):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=color_points,
            s=point_size,
            zorder=2,
            marker="o",
            edgecolors="k",
            linewidths=lw_base,
        )

    delaunay = implicits.delaunay_extend_periodic(points)
    edges = set()
    for triangle in delaunay.simplices:
        for i in range(len(triangle)):
            edge = tuple(sorted([triangle[i], triangle[(i + 1) % len(triangle)]]))
            if edge not in edges:
                edges.add(edge)

    def plot_edges(ax):
        for edge in edges:
            p1, p2 = delaunay.points[edge[0]], delaunay.points[edge[1]]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                "-",
                c=color_edges,
                linewidth=lw_base * 4,
                zorder=1,
            )

    for ax in (ax_triangles, ax_pores):
        plot_edges(ax)

    for ax in (ax_points, ax_triangles, ax_pores):
        default_overview(ax)
        plot_points(ax)
    ax_points.set_title(r"Points $\mathcal{P}$")
    ax_triangles.set_title(r"Triangulation $\mathcal{D}$")
    ax_pores.set_title(r"Indicator function $\mathcal{S}$")

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.07, hspace=0)
    plots.save_pgf(folder_path, "overview_geometry")


def plot_triangle_retractions(parameters, folder_path):
    triangle = [
        np.array([0.3, 0.06]),
        np.array([0.96, 0.95]),
        np.array([0.05, 0.75]),
    ]
    centroid = np.mean(triangle, axis=0)

    def default(ax):
        ax.axis("off")
        ax.add_patch(matplotlib.patches.Polygon(triangle, fill=False))
        ax.set_aspect("equal")

    def colorbar(mappable, ax, ticks):
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation="horizontal",
            shrink=0.7,
            pad=0,
            aspect=15,
        )
        cbar.set_ticks(ticks)
        cbar.update_ticks()

    subdiv = np.linspace(0, 1, plots.subdiv_val)
    levels_01 = np.linspace(0, 1, plots.subdiv_val)
    X0, X1 = np.meshgrid(subdiv, subdiv)
    (
        fig,
        (
            (ax_base, ax_rad, ax_basexp, ax_mat),
            (ax_base_sample, ax_rad_sample, ax_basexp_sample, ax_mat_sample),
        ),
    ) = plots.plot_init(
        to_pgf=True,
        wide=False,
        ncols=4,
        nrows=2,
        gridspec_kw=None,
        aspectratio=1.0,
    )

    # barycentric base image
    base_field = implicits.R(x0=X0, x1=X1, T=triangle)
    base_field_contour = ax_base.contourf(
        X0, X1, base_field, levels=levels_01, cmap=plots.cmap_settings["base"]
    )
    plots.set_rasterized(base_field_contour)
    point_size_centroid = 2
    ax_base.plot(centroid[0], centroid[1], "ko", markersize=point_size_centroid)
    default(ax_base)
    ax_base.set_title(r"$\mathcal{R}$")
    ticks = [0, 0.5, 1]
    colorbar(base_field_contour, ax_base, ticks=ticks)

    plots.noticks_equalaspect(ax_base_sample)
    base_sample_field = implicits.triangulation(
        n=None,
        d=None,
        φ=None,
        β=None,
        resolution=parameters["resolution"],
        seed=parameters["seed"],
        num_points=parameters["num_points"],
        upsampling=parameters["upsampling"],
        triangle_function=implicits.R,
    )
    ax_base_sample.imshow(
        base_sample_field,
        cmap=plots.cmap_settings["base"],
        interpolation=plots.imshowinterp,
    )

    # radial image
    α = 0.2
    rad_field = implicits.E(x0=X0, x1=X1, T=triangle, n=parameters["n"], α=α, d=parameters["d"])
    rad_field = np.where(base_field > 0, rad_field, 0)
    base_rad_contour = ax_rad.contourf(
        X0, X1, rad_field, levels=levels_01, cmap=plots.cmap_settings["rad"]
    )
    plots.set_rasterized(base_rad_contour)
    default(ax_rad)
    ax_rad.set_title(r"$\mathcal{E}$")
    colorbar(base_rad_contour, ax_rad, ticks=ticks)

    plots.noticks_equalaspect(ax_rad_sample)
    radial_sample_field = implicits.triangulation(
        n=parameters["n"],
        d=parameters["d"],
        φ=None,
        β=None,
        resolution=parameters["resolution"],
        seed=parameters["seed"],
        num_points=parameters["num_points"],
        upsampling=parameters["upsampling"],
        triangle_function=implicits.E,
    )
    ax_rad_sample.imshow(
        radial_sample_field,
        cmap=plots.cmap_settings["rad"],
        interpolation=plots.imshowinterp,
    )
    ax_rad_sample.set_title(
        r"$\quad n=" + str(parameters["n"]) + r"\quad d=" + str(parameters["d"]) + "$"
    )

    # base-exponent image
    φ = (parameters["φ"][0] + parameters["φ"][1]) / 2
    basexp_field = implicits.U(
        x0=X0, x1=X1, T=triangle, n=parameters["n"], α=α, d=parameters["d"], φ=φ
    )
    basexp_contour = ax_basexp.contourf(
        X0,
        X1,
        basexp_field,
        levels=levels_01,
        cmap=plots.cmap_settings["basexp"],
    )
    plots.set_rasterized(basexp_contour)
    default(ax_basexp)
    ax_basexp.set_title(r"$\mathcal{U}$")
    plots.noticks_equalaspect(ax_basexp_sample)
    basexp_sample_field = implicits.triangulation(
        n=parameters["n"],
        d=parameters["d"],
        φ=φ,
        β=None,
        resolution=parameters["resolution"],
        seed=parameters["seed"],
        num_points=parameters["num_points"],
        upsampling=parameters["upsampling"],
        triangle_function=implicits.U,
    )
    ax_basexp_sample.imshow(
        basexp_sample_field,
        cmap=plots.cmap_settings["basexp"],
        interpolation=plots.imshowinterp,
    )
    ax_basexp_sample.set_title(r"$\varphi=" + str(φ) + "$")
    colorbar(basexp_contour, ax_basexp, ticks=ticks)

    # solid fraction image
    β = np.round((parameters["β"][0] + parameters["β"][1]) / 2, 1)
    mat_field = implicits.S(
        x0=X0,
        x1=X1,
        T=triangle,
        n=parameters["n"],
        α=α,
        d=parameters["d"],
        φ=φ,
        β=β,
    )
    mat_contour = ax_mat.contourf(X0, X1, mat_field, levels=[0, 0.5, 1], cmap=plots.cmap_bw)
    plots.set_rasterized(mat_contour)
    default(ax_mat)
    ax_mat.set_title(r"$\mathcal{S}$")
    plots.noticks_equalaspect(ax_mat_sample)
    mat_sample_field = implicits.triangulation(
        n=parameters["n"],
        d=parameters["d"],
        φ=φ,
        β=β,
        resolution=parameters["resolution"],
        seed=parameters["seed"],
        num_points=parameters["num_points"],
        upsampling=parameters["upsampling"],
    )
    ax_mat_sample.imshow(
        mat_sample_field,
        cmap=plots.cmap_settings["basexp"],
        interpolation=plots.imshowinterp,
    )
    ax_mat_sample.set_title(r"$\beta=" + str(β) + "$")
    colorbar(mat_contour, ax_mat, ticks=[0, 1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.04, hspace=0.4)
    plots.save_pgf(folder_path, "retraction")

    # retraction gallery
    color_solid = "k"
    color_void = "white"
    φ_gal = np.round(np.linspace(parameters["φ"][0], parameters["φ"][1], 6), 2)
    β_gal = np.round(np.linspace(parameters["β"][0], parameters["β"][1], 4), 2)
    num_cols_ret_gal = len(φ_gal)
    num_rows_ret_gal = len(β_gal)
    fig, axes_gal = plots.plot_init(
        to_pgf=True, wide=False, ncols=num_cols_ret_gal, nrows=num_rows_ret_gal
    )
    for c in range(num_cols_ret_gal):
        for r in range(num_rows_ret_gal):
            ret_field = implicits.S(
                x0=X0,
                x1=X1,
                T=triangle,
                n=parameters["n"],
                α=α,
                d=parameters["d"],
                φ=φ_gal[c],
                β=β_gal[r],
            )
            plots.set_rasterized(
                axes_gal[r][c].contourf(X0, X1, ret_field, levels=[0, 0.5, 1], cmap=plots.cmap_bw)
            )
            if c == 0:
                axes_gal[r][c].text(
                    -0.17,
                    0.5,
                    r"$\beta=" + str(β_gal[r]) + "$",
                    rotation=90,
                    va="center",
                )
            if r == 0:
                axes_gal[r][c].set_title(r"$\varphi=" + str(φ_gal[c]) + "$")
            default(axes_gal[r][c])
    legend_handles = [
        matplotlib.patches.Rectangle((0, 0), 1, 1, color=color_solid, ec="black", lw=0.5),
        matplotlib.patches.Rectangle((0, 0), 1, 1, color=color_void, ec="black", lw=0.5),
    ]
    legend_labels = ["Solid", "Void"]
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        ncol=num_cols_ret_gal,
        bbox_to_anchor=(0.5, -0.1),
    )
    plt.subplots_adjust(left=0.05, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plots.save_pgf(folder_path, "retraction_gallery", pad_inches=0.1)


def main():
    (parameters, folder_path) = common.parse_parameters()

    print("* Save parameters to latex newcommands")
    with open(os.path.join(folder_path, "parameters.tex"), "w") as texfile:
        results_dict = {}
        for parameter in parameters:
            value = parameters[parameter]
            if isinstance(value, int):
                exp2 = common.get_exp_power_two(value)
                if exp2 is not None:
                    if exp2 > 10:
                        value = r"2^{" + str(exp2) + "}"
            common.write_value_tex_and_dict(texfile, parameter, value, results_dict)
        pppval = parameters["resolution"] ** 2 / parameters["num_points"]
        assert pppval.is_integer()
        pppval = int(pppval)
        exp2 = common.get_exp_power_two(pppval)
        if exp2 is not None:
            pppval = r"2^{" + str(exp2) + "}"
        common.write_value_tex_and_dict(texfile, "ppp", pppval, results_dict)

    print("* Plotting overview geometry (figure 2)")
    plot_overview_geometry(parameters, folder_path)

    print("* Plotting figures overview parameters geometry (figures 3 and 4)")
    plot_triangle_retractions(parameters, folder_path)


if __name__ == "__main__":
    main()
