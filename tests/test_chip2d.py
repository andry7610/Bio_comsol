"""Locks for chip2d (IoLit stage 7, part 1)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biological.chip2d import Chip2D


def test_pore_geometry():
    chip = Chip2D()  # 200x100, dx=dy=10 nm, pore 10x10 nodes
    assert int(chip.pore.sum()) == 100
    assert int(chip.wall.sum()) == 900
    assert int(chip.membrane.sum()) == 1000
    assert int(chip.bulk.sum()) == 19000


def test_linear_profile_full_height_pore():
    chip = Chip2D(r_pore_nm=500.0)  # pore spans full height -> no membrane wall
    chip.solve_phi()
    x = np.linspace(0.0, chip.Lx, chip.nx)
    exact = 1.0 - x / chip.Lx
    err = float(np.max(np.abs(chip.phi[:, chip.ny // 2] - exact)))
    assert err < 1e-6, "linear profile err %.2e" % err


def test_potential_bounds_and_current_sign():
    chip = Chip2D()
    chip.solve_phi()
    assert chip.phi.min() >= -1e-9
    assert chip.phi.max() <= 1.0 + 1e-9
    assert chip.pore_current() > 0.0


def test_flux_conservation():
    chip = Chip2D()
    chip.solve_phi()
    i1 = chip.column_current(chip.nx // 4)
    i2 = chip.column_current(chip.nx // 2)
    i3 = chip.column_current(3 * chip.nx // 4)
    assert abs(i1 - i2) / abs(i1) < 0.02, "flux leak at membrane: %.3f%%" % (100 * abs(i1 - i2) / abs(i1))
    assert abs(i1 - i3) / abs(i1) < 0.02


def test_membrane_blocks_bulk_current():
    chip = Chip2D()
    chip.solve_phi()
    i0 = int(np.flatnonzero(chip.membrane.any(axis=1))[0])
    i_total = chip.column_current(i0)
    i_pore = chip.pore_current()
    assert abs(i_total - i_pore) <= 1e-12 + 1e-9 * abs(i_pore)


def test_hall_benchmark_2d():
    chip = Chip2D()
    chip.solve_phi()
    rep = chip.report()
    assert rep["rel_err"] < 0.15, "Hall 2D err %.1f%%" % (100 * rep["rel_err"])
