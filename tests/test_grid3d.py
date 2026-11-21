# test_grid3d.py
# Замки Ступени 6, часть 1: геометрия 3D-сетки.
# Философия та же, что у 2D: аналитика вместо "выглядит нормально".

import time

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from biological.grid3d import GridGraph3D


def test_edge_count_formula():
    g = GridGraph3D(4, 5, 6)
    E = (4 - 1) * 5 * 6 + 4 * (5 - 1) * 6 + 4 * 5 * (6 - 1)
    assert len(g.edges) == E
    assert g.N == 4 * 5 * 6


def test_index_coords_bijection():
    g = GridGraph3D(3, 4, 5)
    seen = set()
    for x in range(3):
        for y in range(4):
            for z in range(5):
                idx = g.index(x, y, z)
                assert 0 <= idx < g.N
                assert idx not in seen
                seen.add(idx)
                assert g.coords(idx) == (x, y, z)
    assert sorted(seen) == list(range(g.N))


def test_laplacian_symmetry_and_row_sums():
    g = GridGraph3D(4, 4, 4)
    L = g.laplacian
    diff = abs(L - L.T)
    assert diff.max() < 1e-12
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    assert np.abs(row_sums).max() < 1e-12
    # степени: 8 углов по 3, рёбра куба по 4, грани по 5, внутренность по 6
    deg = g.degrees
    assert (deg == 3).sum() == 8
    assert (deg == 6).sum() == 8  # для 4^3 внутренность = 2^3 = 8 узлов


def test_linear_field_harmonic_on_interior():
    # phi = x гармонично ТОЛЬКО внутри: на граничных узлах степень < 6,
    # соседа снаружи нет — остаток там честно ненулевой (не баг).
    g = GridGraph3D(5, 5, 5)
    xs = np.array([g.coords(i)[0] for i in range(g.N)], dtype=float)
    resid = g.laplacian @ xs
    interior = g.interior_nodes()
    assert np.abs(resid[interior]).max() < 1e-12
    bnd = [i for i in range(g.N) if i not in set(interior)]
    assert np.abs(resid[bnd]).max() > 0.5


def test_dirichlet_linear_profile_exact():
    # Ванны Дирихле на гранях x: phi=0 (lo), phi=-1.8 (hi).
    # Точное решение внутри — линейный профиль, невязка < 1e-12.
    g = GridGraph3D(6, 6, 6)
    lo, hi = g.face_nodes(0, 0), g.face_nodes(0, 1)
    dirichlet = set(lo) | set(hi)
    unknown = [i for i in range(g.N) if i not in dirichlet]
    phi = np.zeros(g.N)
    phi[hi] = -1.8
    # Неизвестны ВСЕ узлы, кроме граней x. На гранях y/z — естественная
    # граница (нулевой поток): линейный по x профиль ей точно удовлетворяет.
    rhs = -(g.laplacian @ phi)[unknown]
    A = g.laplacian[unknown][:, unknown].tocsc()
    phi_u = spla.spsolve(A, rhs)
    phi_full = phi.copy()
    phi_full[unknown] = phi_u
    xs = np.array([g.coords(i)[0] for i in range(g.N)], dtype=float)
    expected = -1.8 * xs / 5.0
    assert np.abs(phi_full - expected).max() < 1e-12


def test_faces_cover_boundary_exactly():
    # Шесть граней куба 6^3 покрывают ровно 152 граничных узла,
    # внутренность (64 узла) с ними не пересекается.
    g = GridGraph3D(6, 6, 6)
    faces = [g.face_nodes(a, s) for a in (0, 1, 2) for s in (0, 1)]
    for f in faces:
        assert len(f) == 36
        assert len(set(f)) == len(f)
    all_bnd = set().union(*[set(f) for f in faces])
    interior = set(g.interior_nodes())
    assert len(all_bnd) == 152
    assert not (all_bnd & interior)


def test_degenerate_grids():
    # (1,1,5): цепочка из 5 узлов вдоль z.
    g = GridGraph3D(1, 1, 5)
    assert g.N == 5
    assert len(g.edges) == 4
    L = g.laplacian.toarray()
    assert L[0, 0] == 1.0 and L[0, 1] == -1.0
    assert np.abs(L.sum(axis=1)).max() < 1e-12
    assert g.face_nodes(2, 0) == [0] and g.face_nodes(2, 1) == [4]
    assert g.interior_nodes() == []

    # (1,1,1): один узел, пустые рёбра, вырожденный лапласиан — не падаем.
    g1 = GridGraph3D(1, 1, 1)
    assert g1.N == 1 and len(g1.edges) == 0
    assert g1.laplacian.shape == (1, 1)
    assert g1.laplacian.toarray()[0, 0] == 0.0
    assert g1.face_nodes(0, 0) == [0] and g1.face_nodes(0, 1) == [0]
    assert g1.interior_nodes() == []


def test_connectivity():
    g = GridGraph3D(4, 3, 5)
    n, _ = connected_components(g.adjacency, directed=False)
    assert n == 1
    # (1,1,1) — тривиально связен (один компонент из одного узла)
    g1 = GridGraph3D(1, 1, 1)
    n1, _ = connected_components(g1.adjacency, directed=False)
    assert n1 == 1


def test_build_50cubed_under_5s():
    t0 = time.perf_counter()
    g = GridGraph3D(50, 50, 50)
    dt = time.perf_counter() - t0
    assert g.N == 125000
    assert len(g.edges) == 367500
    assert dt < 5.0
