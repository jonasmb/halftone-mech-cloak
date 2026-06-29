import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="Rasterization of '<matplotlib.contour.QuadContourSet object at .*>' will be ignored",
)

# Global default plot settings
cmap_bw = matplotlib.colors.ListedColormap(["white", "black"])
fontsize = 7
scatter_dot_size = 4
linewidth = 0.7
linewidthscatter = 0.01
global_num_ticks_cb = 6
nbins = 4
dpi = 900
subdiv_val = 256
poisson_y_off = 0.01
imshowinterp = "hanning"

cmap_settings = {
    "β": "inferno",
    "φ": "viridis",
    "young_iso": "Blues",
    "solid_fraction": "Greens",
    "poisson_iso": "Oranges",
    "poisson_deviation": "GnBu",
    "mat": "gray_r",
    "base": "afmhot_r",
    "rad": "pink_r",
    "basexp": "bone_r",
    "poisson_ratio": "cividis_r",
    "von_mises_stress": "magma",
    "spectral": "bwr",
}

labels = {
    "bulk_iso": r"Bulk modulus $\kappa$",
    "bulk_iso_short": r"$\kappa$",
    "bulk_iso_hs_up": r"$\kappa_{up}^{HS}$",
    "shear_iso": r"Shear modulus $\mu$",
    "shear_iso_short": r"$\mu$",
    "shear_iso_hs_up": r"$\mu_{up}^{HS}$",
    "young_iso": r"Young's modulus $E$",
    "young_iso_short": r"$E$",
    "target_young": r"$E_{t}$",
    "young_iso_hs_up": r"$E_{up}^{HS}$",
    "poisson_iso": r"Poisson's ratio $v$",
    "poisson_iso_short": r"$v$",
    "dev_iso": r"Deviation of isotropy $\delta_{iso}$",
    "dev_iso_short": r"$\delta_{iso}$",
    "num_points": r"Number of points $k$",
    "num_points_short": r"$k$",
    "resolution": r"Resolution $n$",
    "resolution_short": r"$n$",
    "poisson_ratio": r"Poisson's ratio solid phase $v_{s}$",
    "poisson_ratio_short": r"$v_{s}$",
    "young_max": r"Young's modulus solid phase $E_{s}$",
    "expectation": r"$\mathbb{E}[\mathcal{S}]$",
    "variance": r"$\mathbb{V}[\mathcal{S}]$",
    "pixels_per_point": r"$n_{k}$",
    "solid_fraction": r"Solid fraction $\tau$",
    "solid_fraction_short": r"$\tau$",
    "target_solid_fraction": "Target solid fraction",
    "β_long": r"Parameter $\beta$",
    "β": r"$\beta$",
    "β_symb": r"\beta",
    "φ_long": r"Parameter $\varphi$",
    "φ": r"$\varphi$",
    "φ_symb": r"\varphi",
    "horizontal_load": "Horizontal tension",
    "horizontal_load_short": "Horizontal",
    "vertical_load": "Vertical tension",
    "vertical_load_short": "Vertical",
    "shear_load": "Shearing",
    "two_phases": "Two phases",
    "hokusai": "Hokusai wave",
    "foam": "Foam",
    "example_qr": "QR code",
    "floral_design": "Floral design",
    "julia_set": "Julia set",
    "disp_error": r"$u_{\text{err}}$",
}

tol_muted = [
    "#332288",  # indigo
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]


def plot_init(
    to_pgf,
    wide,
    ncols=1,
    nrows=1,
    gridspec_kw=None,
    aspectratio=1,
    layoutmosaic=None,
    subplot_opts={},
):
    width_in_inches = (504.0 if wide else 240.0) / 72.27
    figsize = (width_in_inches, nrows * width_in_inches / ncols * aspectratio)
    plt.close("all")

    matplotlib.rcParams.update(
        {
            "image.origin": "lower",
            "font.size": fontsize,
            "axes.labelsize": fontsize - 1,
            "xtick.labelsize": fontsize - 2,
            "ytick.labelsize": fontsize - 2,
            "legend.fontsize": fontsize,
            "axes.titlesize": fontsize,
            "legend.frameon": False,
            "xtick.major.size": 0.6,
            "ytick.major.size": 0.6,
            "xtick.minor.size": 0.3,
            "ytick.minor.size": 0.3,
            "xtick.major.pad": 0.8,
            "ytick.major.pad": 0.4,
            "axes.labelpad": 2,
            "axes.linewidth": 0.2,
            "lines.linewidth": 0.5,
            "patch.linewidth": 0.5,
            "path.simplify": False,
        },
    )

    if to_pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "pgf.texsystem": "pdflatex",
                "text.usetex": True,
                "pgf.rcfonts": False,
                "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}",
            },
        )
    else:
        matplotlib.use("TkAgg")

    fig, axes = (
        plt.subplot_mosaic(layoutmosaic, figsize=figsize, gridspec_kw=gridspec_kw)
        if layoutmosaic
        else plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=figsize,
            gridspec_kw=gridspec_kw,
            dpi=dpi,
            **subplot_opts,
        )
    )
    fig.tight_layout()
    return fig, axes


def save_pgf(folder_path, name, pad_inches=0, formatfig="pgf"):
    plt.savefig(
        os.path.join(folder_path, f"{name}.{formatfig}"),
        format=formatfig,
        bbox_inches="tight",
        dpi=dpi,
        pad_inches=pad_inches,
    )


def set_rasterized(contour):
    contour.set_rasterized(True)


def noticks_equalaspect(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_image_porous_structure(ax, image):
    ax.set_aspect("equal")
    ax.imshow(
        image,
        vmin=0,
        vmax=1,
        cmap=cmap_settings["mat"],
        extent=(0, image.shape[1], 0, image.shape[0]),
    )


def plot_data(
    ax_data,
    ax_all,
    ax_color,
    ax_cmap,
    ax_ylabel,
    ax_top,
    ax_bottom,
    ax_cb,
    solid_fraction,
    x_label_mask=None,
    y_ticks_mask=None,
    x_ticks_mask=None,
):
    for i, ax in enumerate(ax_all):
        if x_label_mask is None or x_label_mask[i]:
            ax.set_xlabel(labels["solid_fraction"])
        if y_ticks_mask and not y_ticks_mask[i]:
            ax.get_yaxis().set_ticklabels([])
        if x_ticks_mask and not x_ticks_mask[i]:
            ax.get_xaxis().set_ticklabels([])
        ax.set_xlim([0, 1])
        ax.set_ylim(bottom=ax_bottom[i], top=ax_top[i])
        ax.set_ylabel(ax_ylabel[i])
        if ax_data[i] is not None:
            ax_scatter = ax.scatter(
                solid_fraction,
                ax_data[i],
                marker=".",
                s=scatter_dot_size,
                c=ax_color[i],
                cmap=ax_cmap[i],
                edgecolors=None,
                linewidths=0,
            )
            if ax_cb[i] is not None:
                cbar = plt.colorbar(
                    ax_scatter,
                    ax=ax,
                    ticks=np.linspace(
                        ax_scatter.get_clim()[0],
                        ax_scatter.get_clim()[1],
                        global_num_ticks_cb,
                    ),
                )
                cbar.ax.yaxis.set_major_formatter(
                    matplotlib.ticker.FormatStrFormatter("%.2f"),
                )
                cbar.ax.yaxis.label.set_rotation(0)
                cbar.set_label(ax_cb[i], labelpad=-2)


def cbarp(fig, mappable, ax, label):
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        ticks=np.linspace(
            mappable.get_clim()[0],
            mappable.get_clim()[1],
            global_num_ticks_cb,
        ),
    )
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    cbar.set_label(label, labelpad=-0.7)


def scatterp(fig, ax, ax_inter, β, φ, βG, φG, c, c_interp, cmap, label):
    vmin, vmax = np.min(c), np.max(c)
    cbarp(
        fig,
        ax.scatter(
            β,
            φ,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=scatter_dot_size * 0.8,
            edgecolors="k",
            linewidths=0.1,
        ),
        ax=ax,
        label=label,
    )
    set_rasterized(
        ax_inter.contourf(
            βG,
            φG,
            c_interp,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=subdiv_val,
        ),
    )
    ax_inter.contour(
        βG,
        φG,
        c_interp,
        colors="k",
        vmin=vmin,
        vmax=vmax,
        levels=20,
        linewidths=0.2,
        linestyles="dashed",
    )
