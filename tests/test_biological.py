"""Тест biological v0.4: квазинейтральность, устойчивость, мембранный потенциал."""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from common.graph_utils import Graph
from biological.diffusion import BiologicalDiffusion


IONS = [
    {"name": "Na", "D": 0.01,  "z": 1,  "c0": 10.0,  "c_left": 145.0, "c_right": 10.0},
    {"name": "K",  "D": 0.02,  "z": 1,  "c0": 100.0, "c_left": 4.0,   "c_right": 140.0},
    {"name": "Cl", "D": 0.015, "z": -1, "c0": 110.0, "c_left": 110.0, "c_right": 110.0},
    {"name": "Ca", "D": 0.005, "z": 2,  "c0": 0.1,   "c_left": 2.5,   "c_right": 0.1},
]


def test_basic_stability():
    """Тест 1: 500 шагов, устойчивость, нет отрицательных концентраций."""
    print("=== Test 1: Basic Stability ===")
    graph = Graph(N=50, p_edge=1.0, seed=42)
    bio = BiologicalDiffusion(
        graph, IONS, dt=1e-3, F_RT=1.0,
        membrane_potential=-1.8
    )

    print(f"Graph: {bio.N} nodes, {len(bio.edges)} edges")
    print(f"Ions: {bio.names}")
    print(f"Membrane potential: {bio.phi_membrane}")
    print(f"Initial quasineutrality error: {bio.quasineutrality_error():.4e}")
    print()

    steps = 500
    max_qn_err = 0.0
    neg_count = 0

    for step in range(steps):
        iters = bio.step()
        qn = bio.quasineutrality_error()
        max_qn_err = max(max_qn_err, qn)
        neg = (bio.c < 0).sum()
        neg_count = max(neg_count, neg)

        if step % 100 == 0 or step == steps - 1:
            print(
                f"Step {step:4d}: Q={bio.total_charge():.4e}, "
                f"QN_err={qn:.4e}, "
                f"range=[{bio.c.min():.3f}..{bio.c.max():.3f}], "
                f"neg={neg}, picard_it={iters}"
            )

    print()
    print(f"Max QN error over run: {max_qn_err:.4e}")
    print(f"Max negative conc:    {neg_count}")
    print(f"Final picard iters:   {bio.last_picard_iters}")

    passed = True
    if max_qn_err > 1.0:
        print("❌ FAIL: Quasineutrality error too large")
        passed = False
    if neg_count > 0:
        print("❌ FAIL: Negative concentrations detected")
        passed = False

    if passed:
        print("✅ Test 1 PASSED")
    return passed


def test_charge_conservation():
    """Тест 2: Сохранение заряда."""
    print("\n=== Test 2: Charge Conservation ===")
    graph = Graph(N=50, p_edge=1.0, seed=42)
    bio = BiologicalDiffusion(
        graph, IONS, dt=1e-3, F_RT=1.0,
        membrane_potential=-1.8
    )

    Q0 = bio.total_charge()
    print(f"Initial charge: {Q0:.6e}")

    charges = [Q0]
    for step in range(200):
        bio.step()
        charges.append(bio.total_charge())

    Q_final = charges[-1]
    dQ = Q_final - Q0
    print(f"Final charge:   {Q_final:.6e}")
    print(f"Delta Q:        {dQ:.6e}")

    dQ_steps = np.abs(np.diff(charges))
    max_jump = np.max(dQ_steps)
    print(f"Max jump per step: {max_jump:.6e}")

    passed = True
    if not np.isfinite(Q_final):
        print("❌ FAIL: Charge is NaN/Inf")
        passed = False
    if max_jump > 1e3:
        print("❌ FAIL: Charge jump too large — instability")
        passed = False

    if passed:
        print("✅ Test 2 PASSED")
    return passed


def test_boltzmann_equilibrium():
    """Тест 3: Стационарное распределение сходится к Больцману."""
    print("\n=== Test 3: Boltzmann Equilibrium ===")
    N = 30
    graph = Graph(N=N, p_edge=1.0, seed=42)

    bio = BiologicalDiffusion(
        graph,
        [
            {"name": "Na", "D": 0.1,  "z": 1,  "c0": 10.0, "c_left": 10.0, "c_right": 0.165},
            {"name": "Cl", "D": 0.15, "z": -1, "c0": 10.0, "c_left": 10.0, "c_right": 60.6},
        ],
        dt=2.0,
        F_RT=1.0,
        membrane_potential=-1.8
    )

    phi = bio.phi
    c_boltz_na = 10.0 * np.exp(-1.0 * (phi - phi[0]))
    c_boltz_cl = 10.0 * np.exp(+1.0 * (phi - phi[0]))

    bio.c_boundary[0, 0] = c_boltz_na[0]
    bio.c_boundary[0, 1] = c_boltz_na[-1]
    bio.c_boundary[1, 0] = c_boltz_cl[0]
    bio.c_boundary[1, 1] = c_boltz_cl[-1]

    bio.c[0, 0] = c_boltz_na[0]
    bio.c[0, -1] = c_boltz_na[-1]
    bio.c[1, 0] = c_boltz_cl[0]
    bio.c[1, -1] = c_boltz_cl[-1]

    print(f"Graph: {N} nodes, linear phi: 0 -> {phi[-1]:.2f}")
    print(f"Na Boltzmann: [{c_boltz_na[0]:.4f} ... {c_boltz_na[-1]:.4f}]")
    print(f"Cl Boltzmann: [{c_boltz_cl[0]:.4f} ... {c_boltz_cl[-1]:.4f}]")

    c_prev = bio.c.copy()
    for step in range(100000):
        bio.step()

        if step % 1000 == 0:
            drift = np.max(np.abs(bio.c - c_prev))
            if step % 20000 == 0:
                print(f"  step {step}: drift={drift:.4e}")
            if drift < 1e-10:
                print(f"  Converged at step {step}")
                break
            c_prev = bio.c.copy()

    interior = slice(1, -1)
    passed = True

    for k, (name, c_anal) in enumerate(zip(["Na", "Cl"], [c_boltz_na, c_boltz_cl])):
        rel_err = np.max(
            np.abs(bio.c[k, interior] - c_anal[interior]) / np.maximum(c_anal[interior], 1e-10)
        )
        print(f"\n{name}:")
        print(f"  Sim:       [{bio.c[k,1]:.4f} ... {bio.c[k,N//2]:.4f} ... {bio.c[k,-2]:.4f}]")
        print(f"  Boltzmann: [{c_anal[1]:.4f} ... {c_anal[N//2]:.4f} ... {c_anal[-2]:.4f}]")
        print(f"  Max Rel Error (interior) = {rel_err:.4%}")
        if rel_err > 0.05:
            passed = False

    if passed:
        print("\n✅ Test 3 PASSED")
    else:
        print("\n❌ Test 3 FAILED")
    return passed


def test_spike_response():
    """Тест 4: Отклик на скачок граничных концентраций."""
    print("\n=== Test 4: Spike Response ===")
    graph = Graph(N=50, p_edge=1.0, seed=42)
    bio = BiologicalDiffusion(
        graph, IONS, dt=1e-3, F_RT=1.0,
        membrane_potential=-1.8
    )

    for _ in range(100):
        bio.step()

    Na_before = bio.c[0, bio.N // 2]

    bio.c_boundary[0, 0] = 200.0
    print(f"Na at center before spike: {Na_before:.4f}")
    print(f"Na c_left after spike:     {bio.c_boundary[0, 0]:.4f}")

    for step in range(200):
        bio.step()
        if step % 50 == 0:
            print(
                f"  step {step}: Na_center={bio.c[0, bio.N//2]:.4f}, "
                f"K_center={bio.c[1, bio.N//2]:.4f}, "
                f"QN_err={bio.quasineutrality_error():.4e}"
            )

    Na_after = bio.c[0, bio.N // 2]
    dNa = Na_after - Na_before
    print(f"\nNa at center after spike: {Na_after:.4f} (delta={dNa:+.4f})")

    passed = True
    if not np.isfinite(bio.c).all():
        print("❌ FAIL: NaN/Inf in concentrations")
        passed = False
    if bio.c.min() < -1e-6:
        print("❌ FAIL: Negative concentrations after spike")
        passed = False
    if abs(dNa) < 1e-6:
        print("❌ FAIL: No response to spike")
        passed = False

    if passed:
        print("✅ Test 4 PASSED")
    return passed


def main():
    results = []
    results.append(("Basic Stability", test_basic_stability()))
    results.append(("Charge Conservation", test_charge_conservation()))
    results.append(("Boltzmann Equilibrium", test_boltzmann_equilibrium()))
    results.append(("Spike Response", test_spike_response()))

    print("\n" + "=" * 50)
    print("=== SUMMARY ===")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:25s} {status}")
    print("=" * 50)

    all_passed = all(p for _, p in results)
    if all_passed:
        print("\n🎉 All tests PASSED!")
    else:
        print("\n⚠️  Some tests FAILED — check output above.")


if __name__ == "__main__":
    main()
