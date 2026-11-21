# tests/test_units.py
# Ступень 1 v1.0: замки контракта единиц (units.py).
# Все внутренние расчёты солвера — безразмерные (F_RT=1.0);
# units.py — адаптер на границе "биология (мВ, мкм) <-> SI (В, м)".

import math
import pytest
import units


class TestConstants:
    def test_codata_values(self):
        assert units.FARADAY == pytest.approx(96485.33212, rel=1e-12)
        assert units.GAS_R == pytest.approx(8.314462618, rel=1e-12)
        assert units.T_BODY == pytest.approx(310.15, rel=1e-12)

    def test_rt_f_derived_not_hardcoded(self):
        # Рабочие связки обязаны вычисляться из констант, не быть зашитыми.
        rtf = units.GAS_R * units.T_BODY / units.FARADAY
        assert units.RT_F == rtf
        assert units.F_RT == 1.0 / rtf
        assert units.RT_F * units.F_RT == pytest.approx(1.0, abs=1e-15)

    def test_rt_f_nernst_slope_37c(self):
        # Наклон Нернста при 37 C: RT/F = 26.7267 мВ.
        assert units.v_to_mv(units.RT_F) == pytest.approx(26.7267, abs=1e-4)

    def test_nernst_potassium_sanity(self):
        # E_K при [K+]_out/[K+]_in = 5/140 при 37 C ~ -89 мВ (учебник: ~-90).
        e_k = units.v_to_mv(units.RT_F * math.log(5.0 / 140.0))
        assert e_k == pytest.approx(-89.06, abs=0.1)


class TestMvV:
    def test_scales(self):
        assert units.MV_TO_V == 1e-3
        assert units.V_TO_MV == 1e3

    def test_zero_exact(self):
        assert units.mv_to_v(0.0) == 0.0
        assert units.v_to_mv(0.0) == 0.0

    def test_roundtrip(self):
        for v in (-80.0, -70.5, -1.8, 0.0, 12.3, 55.0):
            assert units.v_to_mv(units.mv_to_v(v)) == pytest.approx(
                v, rel=1e-12, abs=1e-12)

    def test_roundtrip_not_bitexact_documented(self):
        # Оговорка из шапки units.py: туда-обратно не бит-в-бит.
        # Замок фиксирует ДОКУМЕНТИРОВАННОЕ поведение: расхождение есть,
        # но крошечное (< 1e-13 мВ). Не менять на == !
        rng = __import__("numpy").random.default_rng(42)
        vals = rng.uniform(-100, 100, 50_000)
        back = units.v_to_mv(units.mv_to_v(vals))
        diff = abs(back - vals)
        assert diff.max() < 1e-13
        # и хотя бы одно значение реально не совпало бит-в-бит:
        assert (diff > 0).sum() > 0


class TestUm:
    def test_scales(self):
        assert units.UM_TO_M == 1e-6
        assert units.M_TO_UM == 1e6

    def test_roundtrip_and_scale(self):
        # 10 мкм -> 1e-5 м -> обратно 10 мкм.
        meters = 10.0 * units.UM_TO_M
        assert meters == pytest.approx(1e-5)
        assert meters * units.M_TO_UM == pytest.approx(10.0)
