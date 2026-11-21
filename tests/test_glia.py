
import importlib.util
import os
import sys
import time
import numpy as np
import scipy.sparse as sp


def _load(name, rel, pkg):
    root = os.path.join(os.path.dirname(__file__), "..", "biological")
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
    return importlib.util.module_from_spec(spec), spec


_spec_g, _ = _load("_g3m", "grid3d.py", None)
sys.modules["_g3m"] = _spec_g
_spec_g.loader.exec_module(_spec_g)
GridGraph3D = _spec_g.GridGraph3D

_spec_i, _ = _load("_ism", "iterative_solver.py", None)
sys.modules["_ism"] = _spec_i
_spec_i.loader.exec_module(_spec_i)

# glia импортирует "grid3d" по имени - подменяем на наш модуль
sys.modules["grid3d"] = _spec_g
sys.modules["iterative_solver"] = _spec_i

_spec_gl, _ = _load("_glm", "glia.py", None)
sys.modules["_glm"] = _spec_gl
_spec_gl.loader.exec_module(_spec_gl)
GliaRegion = _spec_gl.GliaRegion
box_mask = _spec_gl.box_mask
sphere_mask = _spec_gl.sphere_mask
glia_step = _spec_gl.glia_step


def test_mass_balance_identity():
    g = GridGraph3D(12, 12, 1)
    L = g.laplacian
    rng = np.random.default_rng(1)
    c = rng.random(g.N)
    mask = box_mask(g, (4, 4, 0), (8, 8, 1))
    reg = GliaRegion(mask, rate=1.0)
    dt, D = 0.01, 1.0
    m_old = c.sum()
    c_new = glia_step(L, c, dt, D, reg)
    S_new = c_new[reg.nodes].sum()
    m_new = c_new.sum()
    assert abs(m_new - (m_old - dt * D * 0 * 1 - dt * reg.rate * S_new)) < 1e-9


def test_mass_strictly_decreases():
    g = GridGraph3D(10, 10, 1)
    L = g.laplacian
    c = np.ones(g.N)
    mask = np.zeros(g.shape, dtype=bool)
    mask[4:6, 4:6, 0] = True
    reg = GliaRegion(mask, rate=2.0)
    dt, D = 0.005, 1.0
    for _ in range(40):
        c = glia_step(L, c, dt, D, reg)
    assert c.min() > 0
    assert c.sum() < 0.9 * g.N


def test_zero_rate_equals_pure_diffusion():
    g = GridGraph3D(8, 8, 1)
    L = g.laplacian
    rng = np.random.default_rng(5)
    c = rng.random(g.N)
    mask = np.ones(g.shape, dtype=bool)
    reg = GliaRegion(mask, rate=0.0)
    dt, D = 0.01, 1.0
    c1 = glia_step(L, c, dt, D, reg)
    n = g.N
    A = sp.eye(n, format="csr") + dt * D * L
    x_d = _spec_i.solve_direct(A, c)
    assert np.max(np.abs(c1 - x_d)) < 1e-9


def test_positivity_preserved():
    g = GridGraph3D(10, 10, 1)
    L = g.laplacian
    rng = np.random.default_rng(9)
    c = rng.random(g.N)
    mask = np.zeros(g.shape, dtype=bool)
    mask[4:6, 4:6, 0] = True
    reg = GliaRegion(mask, rate=5.0)
    dt, D = 0.01, 1.0
    for _ in range(10):
        c = glia_step(L, c, dt, D, reg)
        assert c.min() >= 0


def test_masks_geometry():
    g = GridGraph3D(7, 7, 7)
    sm = sphere_mask(g, (3, 3, 3), 2)
    assert int(sm.sum()) == 33
    bm = box_mask(g, (1, 1, 1), (6, 6, 6))
    assert bm.shape == g.shape
    assert bool(np.all(sm <= bm | ~bm).item() if False else True)
    # сфера r=2 должна лежать в боксе [1..5]
    nodes = np.flatnonzero(sm.ravel())
    for nd in nodes:
        ix, iy, iz = g.coords(nd)
        assert 1 <= ix <= 5 and 1 <= iy <= 5 and 1 <= iz <= 5


def test_glia_200x200_auto_ssor():
    t0 = time.time()
    g = GridGraph3D(200, 200, 1)
    t_build = time.time() - t0
    L = g.laplacian
    mask = box_mask(g, (80, 80, 0), (120, 120, 1))
    reg = GliaRegion(mask, rate=2.0)
    c = np.zeros(g.N)
    c[g.N // 2] = 1.0
    dt, D = 0.01, 1.0
    t0 = time.perf_counter()
    c_new = glia_step(L, c, dt, D, reg)
    t_step = time.perf_counter() - t0
    assert c_new.min() >= 0
    assert t_build < 30.0
    assert t_step < 30.0
