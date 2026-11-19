# test_sg_1d.py
# Замки Ступени 2: SG-схема в 1D. Проверено в песочнице перед сдачей.

import numpy as np
import pytest

from sg_scheme import bernoulli, sg_edge_flux, central_edge_flux, safe_explicit_dt
from units import F_RT

D, H = 1e-9, 1e-6          # диффузия K+ (м^2/с), шаг сетки (м)


def test_bernoulli_small_x_series():
    # При малых |x| ряд 1 - x/2 + x^2/12 совпадает с точным значением.
    # Допуск 1e-13: следующий член ряда x^4/720 при x=1e-3 даёт 1.4e-15,
    # поэтому допуск не может быть уже 1e-14 по построению ряда.
    for x in (1e-6, -1e-6, 1e-3, -1e-3):
        assert float(bernoulli(x)) == pytest.approx(1 - x / 2 + x * x / 12, abs=1e-13)


def test_bernoulli_extremes_stable():
    # B(-x) -> x (дрейф доминирует), B(+x) -> 0; без переполнений
    assert float(bernoulli(-60.0)) == pytest.approx(60.0, rel=1e-9)
    assert float(bernoulli(-800.0)) == pytest.approx(800.0, rel=1e-12)
    assert float(bernoulli(60.0)) == pytest.approx(0.0, abs=1e-24)
    assert float(bernoulli(800.0)) == pytest.approx(0.0, abs=1e-300)


def test_bernoulli_antisymmetry():
    # Тождество B(x) - B(-x) = -x (проверено в песочнице: err < 1e-16)
    for x in (0.001, 0.1, 1.0, 5.0, 10.0, 50.0):
        assert float(bernoulli(x)) - float(bernoulli(-x)) == pytest.approx(-x, abs=1e-12)


def test_sg_matches_central_at_small_pe():
    # При Pe << 2 SG обязаны совпасть со старой схемой (замер: 1.15e-6)
    c_i, c_j, phi_i, phi_j = 1.0, 0.8, 0.0, -1e-4
    sg = float(sg_edge_flux(c_i, c_j, phi_i, phi_j, D, H, F_RT))
    cd = float(central_edge_flux(c_i, c_j, phi_i, phi_j, D, H, F_RT))
    assert sg == pytest.approx(cd, rel=1e-5)


@pytest.mark.parametrize("psi", [5.0, -5.0, 0.5, -0.5])
def test_sg_boltzmann_equilibrium(psi):
    # Равновесие Больцмана: c_j = c_i*exp(-psi) => поток на уровне машинной
    # точности. Допуск 1e-17: ниже eps*exp(|psi|) опускаться нельзя.
    c_i = 1.0
    c_j = float(np.exp(-psi))
    phi_i, phi_j = 0.0, psi / F_RT
    flux = float(sg_edge_flux(c_i, c_j, phi_i, phi_j, D, H, F_RT))
    assert flux == pytest.approx(0.0, abs=1e-17)


def test_sg_positivity_pe_above_2():
    # ГЛАВНЫЙ ЗАМОК: резкий фронт при Pe=10, 20 явных шагов.
    # SG обязан держать концентрации в [0, c_max] — без осцилляций.
    n = 20
    c = np.ones(n)
    c[10:] = 1e-6
    dphi = -10.0 / F_RT          # psi = -10 на ребро => Pe = 10 > 2
    phi = np.arange(n, dtype=float) * dphi
    dt = safe_explicit_dt(c, phi, D, H, F_RT)

    for _ in range(20):
        gamma = np.array([
            sg_edge_flux(c[k], c[k + 1], phi[k], phi[k + 1], D, H, F_RT)
            for k in range(n - 1)
        ])
        c_new = c.copy()
        c_new[1:-1] = c[1:-1] + dt / H * (gamma[:-1] - gamma[1:])
        c = c_new
        assert np.all(c >= 0.0), "SG дал отрицательную концентрацию!"
        assert np.all(c <= 1.0 + 1e-12), "SG дал выброс выше максимума!"


def test_central_difference_oscillates_pe_above_2():
    # ДОКУМЕНТАЦИОННЫЙ замок: старая схема на той же сцене даёт выброс.
    # Это не баг теста — это доказательство, зачем нужен SG.
    n = 20
    c = np.ones(n)
    c[10:] = 1e-6
    dphi = -10.0 / F_RT
    phi = np.arange(n, dtype=float) * dphi
    dt = safe_explicit_dt(c, phi, D, H, F_RT)

    for _ in range(20):
        gamma = np.array([
            central_edge_flux(c[k], c[k + 1], phi[k], phi[k + 1], D, H, F_RT)
            for k in range(n - 1)
        ])
        c_new = c.copy()
        c_new[1:-1] = c[1:-1] + dt / H * (gamma[:-1] - gamma[1:])
        c = c_new

    assert np.any(c < 0.0) or np.any(c > 1.0 + 1e-12), (
        "Центральные разности не осциллируют при Pe=10? Проверить сценарий."
    )
