import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

import common
import curve_interpolation
import pickling

alias_interpolation = "_interpolation"
interp_method = "cubic"
inter_num_βφ = 1024


def main():
    params, folder_path = common.parse_parameters()
    data = common.load_data(sys.argv[1])
    βs, φs = [np.linspace(params[key][0], params[key][1], num=inter_num_βφ) for key in ["β", "φ"]]
    βG, φG = np.meshgrid(βs, φs)
    interp_points = np.stack([data["β"], data["φ"]], axis=1)
    props = ["young_iso", "poisson_iso", "solid_fraction"]
    interp_βφ = {
        p: scipy.interpolate.griddata(interp_points, data[p], (βG, φG), method=interp_method)
        for p in props
    }
    data_interp = {
        "βG": βG,
        "φG": φG,
        "interp_βφ": interp_βφ,
        "E_target_spans": [],
        "interp_points": interp_points,
    }
    for E_target in params["E_targets"]:
        print("* Target Young E_t = " + str(E_target))
        paths = plt.contour(βG, φG, interp_βφ["young_iso"], levels=[E_target]).get_paths()
        assert paths
        vertices = paths[0].vertices
        if len(vertices) > 0:
            solid_fraction = scipy.interpolate.griddata(
                interp_points,
                data["solid_fraction"],
                vertices,
                method=interp_method,
            )
            assert solid_fraction.size > 0
            span_solid_fraction = np.ptp(solid_fraction)
            print("\t- span_solid_fraction = " + str(span_solid_fraction))
            data_interp["E_target_spans"].append(
                {
                    "span_solid_fraction": span_solid_fraction,
                    "young": E_target,
                    "vertices": vertices,
                    "solid_fraction_range": [
                        np.min(solid_fraction),
                        np.max(solid_fraction),
                    ],
                    "solid_fraction": solid_fraction,
                }
            )
    pickling.dump_samples(
        sys.argv[1],
        data_interp,
        prealias=alias_interpolation,
    )

    for elem in range(len(data_interp["E_target_spans"])):
        data_curve_interp_elem = curve_interpolation.curve_fitting_βφ(data_interp, elem=elem)
        print(data_curve_interp_elem)
        curve_interpolation.save_curve_interp(folder_path, data_curve_interp_elem)


if __name__ == "__main__":
    main()
