#!/usr/bin/env python3
"""Regression tests for analytic-vs-IPC tangential-drive units."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from novbts.research.groundtruth.hertz_mindlin import (  # noqa: E402
    NU,
    bonded_layer_contact_radius,
    bonded_layer_tangential_compliance_factor,
    hertz_bonded_layer_field,
    hertz_scalars,
    hertz_mindlin_field,
)


def params_for_drive_ratios(g: np.ndarray, *, shear_scale: float) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    mu = 0.6
    params = np.zeros((len(g), 9), dtype=np.float64)
    params[:, 2] = 3e-4
    params[:, 3] = 4e-3
    params[:, 4] = g * mu * shear_scale
    params[:, 6] = mu
    params[:, 7] = 1e5
    return params


def test_explicit_ipc_scale_recovers_all_slip_modes() -> None:
    g = np.array([0.0, 0.2, 0.7, 1.1])
    params = params_for_drive_ratios(g, shear_scale=1e-3)
    coords = np.array([[0.0, 0.0]], dtype=np.float64)

    _, mode = hertz_mindlin_field(params, coords, shear_scale=1e-3)

    assert mode.tolist() == [0, 1, 2, 3]


def test_default_scale_preserves_analytic_generator_convention() -> None:
    g = np.array([0.0, 0.2, 0.7, 1.1])
    params = params_for_drive_ratios(g, shear_scale=1.0)
    coords = np.array([[0.0, 0.0]], dtype=np.float64)

    _, mode = hertz_mindlin_field(params, coords)

    assert mode.tolist() == [0, 1, 2, 3]


def test_invalid_shear_scale_is_rejected() -> None:
    params = params_for_drive_ratios(np.array([0.2]), shear_scale=1.0)
    coords = np.array([[0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="shear_scale"):
        hertz_mindlin_field(params, coords, shear_scale=0.0)


def test_production_mode_counts_when_dataset_is_available() -> None:
    path = ROOT / "data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz"
    if not path.exists():
        pytest.skip("production IPC dataset is not available")

    with np.load(path, allow_pickle=True) as data:
        _, mode = hertz_mindlin_field(
            data["params"], data["coords"], shear_scale=1e-3
        )
        stored_mode = np.asarray(data["mode"], dtype=np.int32)

    assert np.array_equal(mode, stored_mode)
    assert np.bincount(mode, minlength=4).tolist() == [425, 712, 895, 488]


def test_mindlin_amplitude_is_the_closed_form_rigid_shift() -> None:
    """The modern field contains no fitted amplitude or radial Gaussian push."""
    g, shear_scale, mu = 0.7, 1e-3, 0.6
    params = params_for_drive_ratios(np.array([g]), shear_scale=shear_scale)
    coords = np.array([[0.0, 0.0]], dtype=np.float64)
    disp, _ = hertz_mindlin_field(params, coords, shear_scale=shear_scale)
    a, P, *_ = hertz_scalars(params[:, 2], params[:, 3], params[:, 7])
    G = params[:, 7] / (2.0 * (1.0 + NU))
    expected = (
        3.0 * mu * P * (2.0 - NU) / (16.0 * G * a)
        * (1.0 - (1.0 - g) ** (2.0 / 3.0))
    )
    assert np.isclose(disp[0, 0, 0], expected[0], rtol=1e-6)
    assert disp[0, 0, 1] == 0.0


def test_bonded_layer_is_stiffer_than_the_half_space() -> None:
    params = params_for_drive_ratios(np.array([0.0]), shear_scale=1e-3)
    a_h, *_ = hertz_scalars(params[:, 2], params[:, 3], params[:, 7])
    a_layer = bonded_layer_contact_radius(a_h, params[:, 8], thickness_m=3e-3)
    assert 0.0 < a_layer[0] < a_h[0]
    coords = np.array([[0.0, 0.0]], dtype=np.float64)
    half, _ = hertz_mindlin_field(params, coords, shear_scale=1e-3)
    layer, _ = hertz_bonded_layer_field(params, coords, shear_scale=1e-3)
    assert abs(layer[0, 0, 2]) < abs(half[0, 0, 2])


def test_bonded_layer_tangential_compliance_has_the_two_stiffness_limits() -> None:
    """The citable bonded-layer amplitude closure stiffens monotonically."""
    h = 3e-3
    x = np.array([1e-9, 0.1, 0.3, 0.7, 2.0, 1e6])
    factor = bonded_layer_tangential_compliance_factor(x * h, thickness_m=h)
    alpha = np.pi * (2.0 - NU) / 8.0

    assert np.all(np.diff(factor) < 0.0)
    assert np.isclose(factor[0], 1.0, rtol=1e-8)
    # At a/h -> infinity, the stiffness ratio is alpha*a/h, the bonded
    # thin-layer limit of Burger et al. (2023), Eq. (65).
    assert np.isclose(1.0 / factor[-1] / x[-1], alpha, rtol=1e-3)
