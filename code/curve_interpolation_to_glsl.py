import os
import sys

import common
import curve_interpolation
import interpolation
import pickling


def to_poly_glsl(data_curve_fit, name_func):
    coeffs = [float(x) for x in data_curve_fit["popt"].strip("[]").split()]
    yini = str(data_curve_fit["yini"])
    # Horner's method
    expr = str(yini)
    for coeff in coeffs:
        expr += "+x*(" + str(coeff)
    expr += ")" * len(coeffs)
    return f"float {name_func}(float x){{return {expr};}}"


def main():
    params, folder_path = common.parse_parameters()
    data_interp = pickling.load_samples(sys.argv[1], prealias=interpolation.alias_interpolation)
    data_interp_span = data_interp["E_target_spans"][params["E_target_select_halftonings"]]
    target_young = data_interp_span["young"]
    print("Target Young E_t = " + str(target_young))

    data_curve_fit = curve_interpolation.load_data_curve_fit(folder_path, target_young)
    print(data_curve_fit)
    filename_glsl = os.path.join(folder_path, "beta_varphi_polynomials.glsl")
    print("* Writing glsl file at " + str(filename_glsl))
    with open(filename_glsl, "w") as f:
        f.write(
            "// polynomials for beta and varphi with target Young's modulus = "
            + str(target_young)
            + "\n"
        )
        f.write(
            "// solid fraction range "
            + str([data_curve_fit["β"]["xini"], data_curve_fit["β"]["xend"]])
            + "\n"
        )
        f.write("\n" + to_poly_glsl(data_curve_fit["β"], name_func="poly_beta") + "\n")
        f.write("\n" + to_poly_glsl(data_curve_fit["φ"], name_func="poly_varphi") + "\n")


if __name__ == "__main__":
    main()
