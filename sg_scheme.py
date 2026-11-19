# sg_scheme.py
# Ступень 2 (v1.0): схема Шарфеттера-Гуммеля (SG) для потока Нернста-Планка.
# Заменяет центральные разности (E_mid) Бернулли-весами.
# Лечит осцилляции при Pe > 2 — документированную границу паспорта v0.9.
#
# Формула потока через ребро i -> j (на единицу площади):
#     psi   = z * (F/RT) * (phi_j - phi_i)
#     Gamma = (D / h) * [ B(psi) * c_i  -  B(-psi) * c_j ]
# где B(x) = x / (exp(x) - 1) — функция Бернулли, B(x) > 0 при любом x.
#
# Свойства (проверены числом, см. tests/test_sg_1d.py):
#   1) при Pe -> 0 SG переходит в центральные разности (совпадение 1e-6);
#   2) при больцмановском равновесии c_j = c_i*exp(-psi) поток точно ноль;
#   3) положительные коэффициенты -> позитичность концентраций при ЛЮБОМ Pe
#      (при условии ограничения явного шага, см. безопасный dt в тестах).

import numpy as np


def bernoulli(x):
    """Стабильная функция Бернулли B(x) = x / (exp(x) - 1).

    Три режима:
      |x| < 1e-8  -> ряд: 1 - x/2 + x^2/12 (иначе 0/0);
      x > 500     -> x*exp(-x) (иначе переполнение exp);
      иначе       -> x / expm1(x) (expm1 точнее exp(x)-1).
    B(x) > 0 всегда; B(x) -> -x при x -> -inf; B(x) -> 0 при x -> +inf.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    big = x > 500
    rest = ~(small | big)
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    out[big] = x[big] * np.exp(-x[big])
    out[rest] = x[rest] / np.expm1(x[rest])
    return out


def sg_edge_flux(c_i, c_j, phi_i, phi_j, D, h, rt_f, z=1.0):
    """SG-поток иона через ребро i -> j.

    c_i, c_j   — концентрации на узлах ребра (моль/м^3);
    phi_i, phi_j — потенциалы на узлах (В);
    D          — коэффициент диффузии (м^2/с);
    h          — длина ребра (м);
    rt_f       — RT/F (В);
    z          — валентность иона (K+: z = +1).

    Возвращает поток Gamma (моль/(м^2*с)), знак: >0 — от i к j.
    """
    psi = z * rt_f * (phi_j - phi_i)
    return (D / h) * (bernoulli(psi) * c_i - bernoulli(-psi) * c_j)


def central_edge_flux(c_i, c_j, phi_i, phi_j, D, h, rt_f, z=1.0):
    """Старая схема (центральные разности) — ТОЛЬКО для сравнения в тестах.
    Осциллирует при Pe > 2: держим в коде, чтобы замок это доказывал."""
    c_mid = 0.5 * (c_i + c_j)
    dc_dx = (c_j - c_i) / h
    dphi_dx = (phi_j - phi_i) / h
    return -D * (dc_dx + z * rt_f * c_mid * dphi_dx)


def safe_explicit_dt(c, phi, D, h, rt_f, z=1.0, safety=0.1):
    """Безопасный явный шаг: dt <= safety * min_i (c_i / rate_out_i).
    Гарантирует позитичность явного шага SG-схемы. c_i=0 пропускаем."""
    c = np.asarray(c, dtype=float)
    rate_out = np.zeros_like(c)
    psi_edges = z * rt_f * (phi[1:] - phi[:-1])
    b_edges = np.maximum(bernoulli(psi_edges), bernoulli(-psi_edges))
    rate_out[:-1] += (D / h) * b_edges * (c[:-1] > 0)   # отток вправо
    rate_out[1:] += (D / h) * b_edges * (c[1:] > 0)     # отток влево
    active = rate_out > 0
    if not np.any(active):
        return safety / (D / h) if False else safety * h * h / (D * (2 * float(np.max(b_edges)) + 1.0))
    return safety * float(np.min(c[active] / rate_out[active]))
