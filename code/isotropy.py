import numpy as np

# plane stress conversion formulas
# κ -> bulk modulus
# μ -> shear modulus
# E -> Young's modulus
# v -> Poisson's ratio


def κ(E, v):
    return E / (2 * (1 - v))


def μ(E, v):
    return E / (2 * (1 + v))


def E(κ, μ):
    return 4 * κ * μ / (κ + μ)


def v(κ, μ):
    return (κ - μ) / (κ + μ)


def κ_HS_up(tau, E_s, v_s):
    κ_s, μ_s = κ(E_s, v_s), μ(E_s, v_s)
    return (tau * κ_s * μ_s) / ((1 - tau) * κ_s + μ_s)


def μ_HS_up(tau, E_s, v_s):
    κ_s, μ_s = κ(E_s, v_s), μ(E_s, v_s)
    return (tau * κ_s * μ_s) / ((1 - tau) * (κ_s + 2 * μ_s) + κ_s)


def E_HS_up(tau, E_s):
    return (E_s * tau) / (3 - 2 * tau)


def closest_isotropic(C_eff):
    """
    Compute the closest isotropic tensor of an elasticity tensor.

    We assume plane stress and C_eff given in Mandel notation.

    Args:
        C_eff: a 3x3 matrix in Manel notation.

    Returns:
        κ_iso: bulk modulus of the closest isotropic material.
        μ_iso: shear modulus of the closest isotropic material.
        dev_iso: normalized measure of deviation from isotropy.
    """
    assert C_eff.ndim == 2
    assert C_eff.shape[0] == C_eff.shape[1]
    assert C_eff.shape[0] == 3
    dimension = 2
    I = np.identity(C_eff.shape[0])
    J = np.zeros_like(C_eff)
    J[:dimension, :dimension] = 1 / dimension
    K = I - J
    κ_iso = np.sum(C_eff * J) / 2
    μ_iso = np.sum(C_eff * K) / 4
    C_iso_mandel = dimension * κ_iso * J + 2 * μ_iso * K
    norm_C_eff = np.linalg.norm(
        C_eff,
        ord="fro",
    )
    norm_diff_iso_mandel = np.linalg.norm(
        C_eff - C_iso_mandel,
        ord="fro",
    )
    dev_iso = norm_diff_iso_mandel / norm_C_eff
    return κ_iso, μ_iso, dev_iso
