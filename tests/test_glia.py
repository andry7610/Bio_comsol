import sys
import pathlib
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from biological.grid3d import GridGraph3D
from biological.glia import GliaRegion, box_mask, sphere_mask, glia_step
from biological.iterative_solver import solve_direct


def test_mass_balance_identity():
    g = GridGraph3D(12, 12, 1)
    L = g.laplacian
    reg = GliaRegion(box_mask(g, 4, 7, 4, 7, 0, 0), 2.0)
    dt = 0.005
    rng = np.random.default_rng(0)
    c = rng.random(g.N)
    c1, _ = glia_step(c, L, reg, dt)
    m_old = float(c.sum())
    m_new = float(c1.sum())
    assert abs(m_new - (m_old - dt * reg.consumption(c1))) < 1e-8


def test_mass_strictly_decreases():
    g = GridGraph3D(10, 10, 1)
    L = g.laplacian
    reg = GliaRegion(box_mask(g, 4, 5, 4, 5, 0, 0), 2.0)
    rng = np.random.default_rng(1)
    c = rng.random(g.N)
    m_prev = float(c.sum())
    for _ in range(10):
        c, _ = glia_step(c, L, reg, 0.01)
        m = float(c.sum())
        assert m < m_prev - 1e-12
        m_prev = m


def test_zero_rate_equals_pure_diffusion():
    g = GridGraph3D(8, 8, 1)
    L = g.laplacian
    reg = GliaRegion(box_mask(g, 3, 4, 3, 4, 0, 0), 0.0)
    rng = np.random.default_rng(5)
    c = rng.random(g.N)
    dt = 0.01
    c1, _ = glia_step(c, L, reg, dt)
    A = sp.identity(g.N, format="csr") + dt * L
    x = solve_direct(A, c)
    assert np.allclose(c1, x, atol=1e-9)


def test_positivity_preserved():
    g = GridGraph3D(10, 10, 1)
    L = g.laplacian
    reg = GliaRegion(box_mask(g, 4, 5, 4, 5, 0, 0), 2.0)
    c = np.zeros(g.N)
    c[g.index(4, 4, 0)] = 5.0
    for _ in range(10):
        c, _ = glia_step(c, L, reg, 0.005)
        assert c.min() >= -1e-12


def test_masks_geometry():
    g = GridGraph3D(7, 7, 7)
    s = sphere_mask(g, (3, 3, 3), 2)
    assert int(s.sum()) == 33
    assert s[1, 3, 3] and s[3, 3, 5]  # sphere honestly reaches r=2 shell
    b = box_mask(g, 1, 3, 1, 3, 1, 3)
    assert int(b.sum()) == 27


def test_glia_200x200_auto_ssor():
    g = GridGraph3D(200, 200, 1)
    L = g.laplacian
    reg = GliaRegion(box_mask(g, 90, 109, 90, 109, 0, 0), 2.0)
    c = np.ones(g.N)
    dt = 0.005
    m0 = float(c.sum())
    max_iters = 0
    t0 = time.time()
    for _ in range(40):
        c, st = glia_step(c, L, reg, dt)
        max_iters = max(max_iters, st["iters"])
    per_step = (time.time() - t0) / 40
    drop = m0 - float(c.sum())
    assert max_iters <= 50
    assert 0.0005 * m0 < drop < 0.05 * m0
    assert c.min() >= -1e-12
