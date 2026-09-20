def test_boltzmann_equilibrium():
    """Тест 3: Стационарное распределение сходится к Больцману.

    С фиксированным потенциалом φ(x) и граничными условиями Дирихле,
    равновесие: c_k(x) = c_k(0) * exp(-z_k * (φ(x) - φ(0)) * F/RT)
    """
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

    # Больцмановские граничные значения
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

    # Долгая релаксация к равновесию
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

    # Сравнение
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
