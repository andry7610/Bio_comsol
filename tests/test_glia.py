"""Six locks for the glia module. Imports only biological.*"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from biological.grid3d import GridGraph3D
from biological.glia import (GliaRegion, box_mask, sphere_mask,
                             glia_step, glia_series)


def test_mass_balance_identity():
    g = GridGraph3D(12, 12, 1)
    L = g.laplacian
    glia = GliaRegion(box_mask((12, 12, 1), (4, 4, 0), (7, 7, 0)), rate=2.0)
    rng = np.random.default_rng(1)
    c0 = np.abs(rng.random(g.N)) + 0.5
    dt = 0.005
    c_new, _ = glia_step(L, c0, glia, dt)
    m_old = float(c0.sum())
    S_new = float(glia.mask.ravel() @ c_new)
    # exact algebra of the implicit step: m_new = m_old - dt*k*S_new
    assert abs(float(c_new.sum()) - (m_old - dt * 2.0 * S_new)) < 1e-9


def test_mass_strictly_decreases():
    g = GridGraph3D(10, 10, 1)
    L = g.laplacian
    glia = GliaRegion(box_mask((10, 10, 1), (3, 3, 0), (6, 6, 0)), rate=2.0)
    rng = np.random.default_rng(2)
    c = np.abs(rng.random(g.N)) + 0.5
    dt = 0.005
    m_prev = float(c.sum())
    m0 = m_prev
    for _ in range(40):
        c, _ = glia_step(L, c, glia, dt)
        m = float(c.sum())
        assert m < m_prev - 1e-14
        m_prev = m
    ratio = m_prev / m0
    # measured: glia box 16/100 nodes, k=2, dt=0.005, 40 steps -> ~5.7% drop
    assert 0.90 < ratio < 0.999


def test_zero_rate_equals_pure_diffusion():
    g = GridGraph3D(8, 8, 1)
    L = g.laplacian
    glia = GliaRegion(box_mask((8, 8, 1), (2, 2, 0), (5, 5, 0)), rate=0.0)
    rng = np.random.default_rng(3)
    c0 = np.abs(rng.random(g.N)) + 0.5
    dt = 0.01
    n = g.N
    A = sp.identity(n, format="csr") + dt * L.tocsr()
    c_direct = spsolve(A.tocsc(), c0)
    c_glia, _ = glia_step(L, c0, glia, dt, tol=1e-12)
    # threshold tied to CG accuracy, not machine epsilon
    assert np.abs(c_glia - c_direct).max() < 1e-9


def test_positivity_preserved():
    g = GridGraph3D(10, 10, 1)
    L = g.laplacian
    glia = GliaRegion(box_mask((10, 10, 1), (3, 3, 0), (6, 6, 0)), rate=2.0)
    rng = np.random.default_rng(4)
    c = np.abs(rng.random(g.N)) + 0.5
    dt = 0.005
    for _ in range(10):
        c, _ = glia_step(L, c, glia, dt)
        assert c.min() > -1e-12


def test_masks_geometry():
    m = sphere_mask((7, 7, 7), (3, 3, 3), 2.0)
    assert int(m.sum()) == 33
    inner = sphere_mask((7, 7, 7), (3, 3, 3), 1.0)
    assert (m[inner > 0] == 1).all()
    b = box_mask((6, 6, 6), (1, 1, 1), (5, 5, 5))
    assert int(b.sum()) == 125


def test_glia_200x200_auto_ssor():
    t0 = time.time()
    g = GridGraph3D(200, 200, 1)
    t_build = time.time() - t0
    L = g.laplacian
    glia = GliaRegion(box_mask((200, 200, 1), (90, 90, 0), (109, 109, 0)),
                      rate=2.0)
    # impulse right next to glia border: consumption is fast and measurable
    c = np.zeros(g.N)
    c[g.index(89, 100, 0)] = 5.0
    dt = 0.05
    t0 = time.time()
    info = {}
    c10, _ = glia_step(L, c, glia, dt, info=info)
    for _ in range(9):
        c10, st = glia_step(L, c10, glia, dt)
    t_step = (time.time() - t0) / 10
    m0 = float(c.sum())
    drop = 1.0 - float(c10.sum()) / m0
    assert t_build < 2.0
    assert t_step < 0.6
    # measured: 17% drop in 10 steps for adjacent impulse; corridor with margin
    assert 0.05 < drop < 0.40
    assert c10.min() > -1e-12
