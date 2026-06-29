#!/usr/bin/env python
# coding: utf-8

# Full-field homogenization with Janus
# imported from Janus_demo.ipynb

import inspect

import numpy as np
import scipy.sparse.linalg

import janus.fft.serial as fft
import janus.green as green
import janus.material.elastic.linear.isotropic as material

# Computation of the effective stiffness


class LippmannSchwingerOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, E, E0, ν):
        if not np.isscalar(E0):
            raise RuntimeError("Young modulus of reference material must be a scalar value")
        if not np.isscalar(ν):
            raise RuntimeError("Poisson ratio must be a scalar value")
        self.d = E.ndim
        self.E = E
        self.E0 = E0
        self.μ0 = E0 / 2 / (1 + ν)
        self.λ0 = 2 * self.μ0 * ν / (1 - 2 * ν)
        self.ΔE_inv = (1 / (E - E0))[..., None]
        self.ν = ν
        self.α = 1 + ν
        self.β = ν * (1 + ν) if self.d == 2 else ν
        self.field_shape = E.shape + ((self.d * (self.d + 1)) // 2,)
        C0 = material.create(self.μ0, ν, dim=self.d)
        self.Γ0 = green.filtered(C0.green_operator(), E.shape, 1.0, fft.create_real(E.shape))
        super().__init__(np.float64, (np.prod(self.field_shape), np.prod(self.field_shape)))

    def polarization_to_strain(self, τ):
        tr_τ = np.sum(τ[..., : self.d], axis=-1, keepdims=True)
        ε = self.α * τ
        ε[..., : self.d] -= self.β * tr_τ
        return ε * self.ΔE_inv

    def _matvec(self, x):
        τ = x.reshape(self.field_shape)
        η = self.polarization_to_strain(τ) + self.Γ0.apply(τ)
        return η.ravel()

    def _rmatvec(self, x):
        return self._matvec(x)


# To compute the effective stiffness, we must solve the LS equation
# for 3 ($d=2) or 6 ($d=3$) linearly independent macroscopic strains.
# For each load-case, we must extract the relevant column of the matrix
# that represents the effective stiffness tensor.
def effective_stiffness(a, tol, useUmfpack, return_residuals=False):
    b = np.zeros(a.field_shape, dtype=np.float64)
    n = a.field_shape[-1]
    δ = np.zeros((n,), dtype=np.float64)
    δ[: a.d] = 1.0
    ε_macro = np.zeros_like(δ)
    C_eff = np.zeros((n, n), dtype=np.float64)
    spatial_axes = tuple(range(a.d))
    scipy.sparse.linalg.use_solver(useUmfpack=useUmfpack)
    info_global = []
    for i in range(n):
        print("*** i = {0}".format(i))
        ε_macro[i] = 1.0
        tr_ε_macro = np.sum(ε_macro[: a.d])
        b[...] = ε_macro
        residuals = []

        def callback(xk):
            # see https://stackoverflow.com/questions/14243579/print-current-residual-from-callback-in-scipy-sparse-linalg-cg
            frame = inspect.currentframe().f_back
            residual = frame.f_locals["resid"]
            atol = frame.f_locals["atol"]
            residuals.append(residual)
            print("residual = {0},  atol = {1}".format(residual, atol))

        callback_cg = None
        if return_residuals:
            callback_cg = callback
        x, info = scipy.sparse.linalg.cg(a, b.ravel(), rtol=tol, callback=callback_cg, atol=0)
        if info != 0:
            raise RuntimeError()
        τ = x.reshape(a.field_shape)
        τ_avg = np.mean(τ, axis=spatial_axes)
        C_eff[:, i] = a.λ0 * tr_ε_macro * δ + 2 * a.μ0 * ε_macro + τ_avg
        ε_macro[i] = 0.0
        if return_residuals:
            atol = tol * np.linalg.norm(b.ravel())
            info_global.append((residuals, atol))
    if return_residuals:
        return C_eff, info_global
    return C_eff
