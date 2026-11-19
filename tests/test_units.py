# test_units.py
# Замки контракта единиц (Ступень 1). Все сравнения — через pytest.approx.

import math

import pytest

from units import (
    FARADAY, GAS_R, T_BODY, F_RT, RT_F,
    MV_TO_V, V_TO_MV, UM_TO_M, M_TO_UM,
    mv_to_v, v_to_mv,
)


def test_units_roundtrip_rest_potential():
    test_mv = -65.0  # типичный потенциал покоя нейрона
    assert v_to_mv(mv_to_v(test_mv)) == pytest.approx(test_mv, abs=1e-12)


def test_units_roundtrip_spike_amplitude():
    test_mv = 100.0  # амплитуда спайка
    assert mv_to_v(test_mv) == pytest.approx(0.1)
    assert v_to_mv(0.1) == pytest.approx(test_mv)


def test_units_roundtrip_random_sample():
    # Рантрип на произвольном значении — с допуском, не на точном равенстве
    x = 13.5
    assert v_to_mv(mv_to_v(x)) == pytest.approx(x, abs=1e-12)


def test_units_scale_constants():
    assert MV_TO_V == pytest.approx(1e-3)
    assert V_TO_MV == pytest.approx(1e3)
    assert UM_TO_M == pytest.approx(1e-6)
    assert M_TO_UM == pytest.approx(1e6)


def test_units_thermal_voltage_body():
    # RT/F при 37 C (310.15 K) = 26.7267 мВ — золотое число нейронной главы
    expected_rt_f = GAS_R * T_BODY / FARADAY
    assert expected_rt_f == pytest.approx(0.0267267, abs=1e-7)
    assert RT_F == pytest.approx(expected_rt_f)
    assert F_RT == pytest.approx(1.0 / expected_rt_f)
    # Проверка величин: F/RT при 37 C = 37.4158 1/В
    assert F_RT == pytest.approx(37.4158, abs=1e-3)


def test_units_thermal_voltage_room():
    # Сверка с известным значением: RT/F при 25 C (298.15 K) = 25.69 мВ.
    # Точное значение 0.0256926 В; эталон даём с допуском на округление.
    rt_f_25 = GAS_R * 298.15 / FARADAY
    assert rt_f_25 == pytest.approx(0.0256926, abs=1e-6)


def test_units_temperature_body_value():
    assert T_BODY == pytest.approx(310.15)
