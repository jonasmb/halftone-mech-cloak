import os
import subprocess

import matplotlib.pyplot as plt
import numpy
import numpy as np
import PIL
import scipy.spatial


def barycentric(x0, x1, a, b, c):
    v0, v1 = b - a, c - a
    v2_0, v2_1 = x0 - a[0], x1 - a[1]
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20 = v2_0 * v0[0] + v2_1 * v0[1]
    d21 = v2_0 * v1[0] + v2_1 * v1[1]
    denom = d00 * d11 - d01 * d01
    λ1 = (d11 * d20 - d01 * d21) / denom
    λ2 = (d00 * d21 - d01 * d20) / denom
    λ3 = 1.0 - λ1 - λ2
    return (λ1, λ2, λ3)


def R(x0, x1, T, n=None, α=None, d=None, φ=None, β=None):
    return np.prod(np.clip(barycentric(x0, x1, *T), 0, 1), axis=0) * 27


teq = ((0, 0), (1, 0), (1 / 2, np.sqrt(3) / 2))


def v_teq(λ1, λ2, λ3):
    return [
        teq[0][i] * (λ2 - 1 / 3) + teq[1][i] * (λ1 - 1 / 3) + teq[2][i] * (λ3 - 1 / 3)
        for i in range(2)
    ]


def E(x0, x1, T, n, α, d, φ=None, β=None):
    (λ1, λ2, λ3) = barycentric(x0, x1, *T)
    v = v_teq(λ1, λ2, λ3)
    return np.where(
        R(x0, x1, T) > 0,
        np.abs(np.sin(α + 0.5 * n * np.arctan2(v[1], v[0]))) ** d,
        0,
    )


def U(x0, x1, T, n, α, d, φ, β=None):
    return R(x0, x1, T) ** (1 + E(x0, x1, T, n, α, d) * φ)


def S(x0, x1, T, n, α, d, φ, β):
    return np.where(R(x0, x1, T) > 0, np.heaviside(β - U(x0, x1, T, n, α, d, φ), 0), 0)


def Sλ(λ1, λ2, α, n, d, φ, β):
    λ3 = 1 - λ1 - λ2
    R_term = λ1 * λ2 * λ3 * 27
    v = v_teq(λ1, λ2, λ3)
    E_term = np.abs(np.sin(α + 0.5 * n * np.arctan2(v[1], v[0]))) ** d
    U_term = R_term ** (1 + E_term * φ)
    return np.heaviside(β - U_term, 0)


def ccvt_wrapper(num_points, random_seed):
    ccvtbin_path = os.path.join(os.path.dirname(__file__), "ccvt_mod/build/ccvt")
    tempfile_sites = "temp/sites.txt"
    logfile = "temp/ccvt_log.txt"
    assert os.path.exists(ccvtbin_path) or os.path.exists(ccvtbin_path + ".exe")
    if os.path.exists(tempfile_sites):
        os.remove(tempfile_sites)
    cmd_str = ccvtbin_path + " " + str(num_points) + " " + str(random_seed) + " > " + logfile
    subprocess.run(cmd_str, shell=True)
    points = []
    with open(tempfile_sites) as filesites:
        lines = filesites.readlines()
        for line in lines:
            fields = line.strip("\n").split(" ")
            if len(fields) == 2:
                points.append(numpy.array([float(fields[0]), float(fields[1])]))
    assert len(points) == num_points
    return points


def delaunay_extend_periodic(points):
    points_extended = [
        (p + np.array([i, j])) for i in [-1, 0, 1] for j in [-1, 0, 1] for p in points
    ]
    return scipy.spatial.Delaunay(points_extended)


def approx_int_tri(fun, T, int_tri_samples_per_area, int_tri_eval_outside):
    area_T = 0.5 * abs(np.cross(T[1] - T[0], T[2] - T[0]))
    num_samples = max(3, int(np.round(np.sqrt(area_T * int_tri_samples_per_area))))
    val = 0
    s = 0
    for r1 in np.linspace(0, 1, num=num_samples):
        for r2 in np.linspace(0, 1, num=num_samples):
            sq_r1 = np.sqrt(r1)
            λ1 = 1 - sq_r1
            λ2 = sq_r1 * (1 - r2)
            λ3 = sq_r1 * r2
            coord = T[0] * λ1 + T[1] * λ2 + T[2] * λ3
            if not int_tri_eval_outside:
                if coord[0] > 1 or coord[1] > 1 or coord[0] < 0 or coord[1] < 0:
                    continue  # if the sample falls outside the domain [0, 1] do not evaluate
            val += fun(coord)
            s += 1

    if s > 0:
        return val / float(s)
    else:
        # if all samples fall outside the domain [0, 1] return None
        return None


def triangulation(
    n,
    d,
    φ,
    β,
    resolution,
    seed,
    num_points,
    upsampling,
    triangle_function=S,
    func_target_solid_fraction=None,
    func_poly_φ=None,
    func_poly_β=None,
    input_points=None,
    int_tri_samples_per_area=64**2,
    int_tri_eval_outside=False,
):
    assert isinstance(upsampling, int) and upsampling > 0
    resolution_up = resolution * upsampling
    if input_points is None:
        points = ccvt_wrapper(num_points, random_seed=seed)
        delaunay = delaunay_extend_periodic(points)
    else:
        delaunay = scipy.spatial.Delaunay(input_points)

    x0, x1 = np.meshgrid(np.arange(0.5, resolution_up, 1.0), np.arange(0.5, resolution_up, 1.0))
    field = x0 * 0
    α = 0
    np.random.seed(seed)
    for simplex in delaunay.simplices:
        tri_del = np.array(
            [
                delaunay.points[simplex[0]],
                delaunay.points[simplex[1]],
                delaunay.points[simplex[2]],
            ]
        )
        tri_del = tri_del[np.lexsort((tri_del[:, 0], tri_del[:, 1]))]
        tri = resolution_up * tri_del
        tri_min = np.floor(np.min(tri, axis=0)).astype(int)
        tri_max = np.ceil(np.max(tri, axis=0)).astype(int)
        tri_rect_0, tri_rect_1 = [
            np.clip([tri_min[i], tri_max[i]], 0, resolution_up).astype(int) for i in range(2)
        ]
        if tri_rect_0[1] - tri_rect_0[0] > 0 and tri_rect_1[1] - tri_rect_1[0] > 0:
            slices = (
                slice(tri_rect_1[0], tri_rect_1[1]),
                slice(tri_rect_0[0], tri_rect_0[1]),
            )
            tri_slice_x0, tri_slice_x1 = x0[*slices], x1[*slices]
            # compute φ and β of triangle
            if func_target_solid_fraction is not None:
                # grading case
                assert φ is None and β is None
                assert func_poly_φ is not None and func_poly_β is not None
                target_solid_fraction = approx_int_tri(
                    fun=func_target_solid_fraction,
                    T=tri_del,
                    int_tri_samples_per_area=int_tri_samples_per_area,
                    int_tri_eval_outside=int_tri_eval_outside,
                )
                if target_solid_fraction is None:
                    continue
                φ_tri = func_poly_φ(target_solid_fraction)
                β_tri = func_poly_β(target_solid_fraction)
            else:
                # constant case
                φ_tri = φ
                β_tri = β
            tri_slice_field = triangle_function(
                x0=tri_slice_x0,
                x1=tri_slice_x1,
                T=tri,
                n=n,
                α=α,
                d=d,
                φ=φ_tri,
                β=β_tri,
            )
            field[*slices] += tri_slice_field
    if upsampling == 1:
        field_down = field
    elif upsampling > 1:
        field_reshaped = field.reshape(resolution, upsampling, resolution, upsampling)
        field_down = field_reshaped.mean(
            axis=(1, 3)
        )  # average per block of size (upsampling, upsampling)
    assert field_down.shape[0] == field_down.shape[1] == resolution
    return field_down


###################
# Functions to compute distance to triangle boundary
# for comparison purposes only in the paper, they do not have anything to do with S
# Note that the extra parameters of function dist_to_tri_edges
# are redundant and copied for easier rapid integration in the code


def point_seg_dist(x0, x1, a, b):
    ab = b - a
    p = np.dstack([x0, x1])
    t = np.clip(np.sum((p - a) * ab, axis=2) / np.sum(ab * ab), 0, 1)
    closest = a + t[..., np.newaxis] * ab
    return np.linalg.norm(p - closest, axis=2)


def dist_to_tri_edges(x0, x1, T, n, α, d, φ, β):
    # we interpret the input parameter d as the thickness value
    min_dist = np.minimum(
        np.minimum(
            point_seg_dist(x0, x1, T[0], T[1]),
            point_seg_dist(x0, x1, T[1], T[2]),
        ),
        point_seg_dist(x0, x1, T[2], T[0]),
    )
    return np.where(R(x0, x1, T) > 0, np.heaviside(d - min_dist, 0), 0)


###################

if __name__ == "__main__":
    width = 600
    height = 1000
    X0, X1 = np.meshgrid(np.arange(0.5, width, 1.0), np.arange(0.5, height, 1.0))
    T = (
        np.array([0, 0]),
        np.array([width, 0]),
        np.array([width * 3 / 4, height]),
    )
    # Tfun = S(X0, X1, T, n=6, α=0, d=3, φ=0, β=0.5)
    Tfun = dist_to_tri_edges(X0, X1, T, n=None, α=None, d=50, φ=None, β=None)
    print(Tfun.shape)
    plt.imshow(Tfun, origin="lower")
    plt.gca().set_aspect("equal")
    plt.show()
    img = PIL.Image.fromarray((Tfun * 255).astype(np.uint8))
    img.save("triangle_test.png")
