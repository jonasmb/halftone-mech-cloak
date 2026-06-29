import sys

import matplotlib.pyplot as plt
import numpy as np

import common
import isotropy
import plots


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1])
    layoutmosaic = """
    AB
    CC
    """
    fig, axes = plots.plot_init(
        to_pgf=True,
        wide=False,
        aspectratio=1 / 1.5,
        gridspec_kw={"height_ratios": [1.7, 1]},
        layoutmosaic=layoutmosaic,
    )
    ax_bulk, ax_shear, ax_dev_iso = axes["A"], axes["B"], axes["C"]
    plots.plot_data(
        ax_data=[data["bulk_iso"], data["shear_iso"], data["dev_iso"]],
        ax_all=[ax_bulk, ax_shear, ax_dev_iso],
        ax_color=["k", "k", "darkred"],
        ax_cmap=[None, None, None],
        ax_ylabel=[
            plots.labels["bulk_iso"],
            plots.labels["shear_iso"],
            plots.labels["dev_iso"],
        ],
        ax_top=[
            max(
                np.max(data["bulk_iso"]),
                isotropy.κ(E=params["young_max"], v=params["poisson_ratio"]),
            ),
            max(
                np.max(data["shear_iso"]),
                isotropy.μ(E=params["young_max"], v=params["poisson_ratio"]),
            ),
            np.max(data["dev_iso"]) * 1.1,
        ],
        ax_bottom=[0, 0, 0],
        ax_cb=[None, None, None],
        solid_fraction=data["solid_fraction"],
    )
    ax_dev_iso.tick_params(axis="y")
    common.plot_hs_upper_bound(ax=ax_bulk, name="bulk_iso", params=params)
    common.plot_hs_upper_bound(ax=ax_shear, name="shear_iso", params=params)
    plt.subplots_adjust(left=0.1, right=1, bottom=0, top=1, hspace=0.3)
    plots.save_pgf(folder_path, "bulk_shear_deviso", pad_inches=0.02)


if __name__ == "__main__":
    main()
