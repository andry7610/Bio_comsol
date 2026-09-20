"""main.py — Конфиг-драйвенный запуск симуляции Bio-COMSOL.

Использование:
    python main.py config.yaml
    python main.py config.yaml --steps 1000 --output-every 100
"""

import argparse
import numpy as np
import yaml

from network.graph import Graph
from biological.diffusion import BiologicalDiffusion
from biological.neuron import NeuronLayer
from biological.synapse import SynapseLayer


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_graph(cfg):
    g = cfg.get('graph', {})
    return Graph(
        N=g.get('N', 20),
        topology=g.get('topology', 'chain'),
        p_edge=g.get('p_edge'),
        seed=g.get('seed', 42),
        conductivity=g.get('conductivity', 1.0),
        cross_section=g.get('cross_section', 1e-8),
    )


def build_ions(cfg):
    species = cfg.get('ion_species', {})
    ions = []
    for name, params in species.items():
        ions.append({
            'name': name,
            'D': params['D'],
            'z': params['z'],
            'c0': params['c0'],
            'c_left': params.get('c_left', params['c0']),
            'c_right': params.get('c_right', params['c0']),
        })
    return ions


def build_solver(cfg, graph, ions):
    phys = cfg.get('physics_constants', {})
    solver_cfg = cfg.get('solver', {})

    F = phys.get('F', 96485.0)
    R = phys.get('R', 8.314)
    T = phys.get('T', 310.0)
    F_RT = F / (R * T)

    return BiologicalDiffusion(
        graph=graph,
        ions=ions,
        dt=solver_cfg.get('dt', 1e-3),
        F_RT=F_RT,
        membrane_potential=solver_cfg.get('membrane_potential', -1.8),
        picard_tol=solver_cfg.get('picard_tol', 1e-6),
        picard_max_iter=solver_cfg.get('picard_max_iter', 20),
        analyze_every=solver_cfg.get('analyze_every', 0),
        self_consistent=solver_cfg.get('self_consistent', False),
        poisson_lambda=solver_cfg.get('poisson_lambda', 1.0),
        use_newton=solver_cfg.get('use_newton', False),
        newton_tol=solver_cfg.get('newton_tol', 1e-8),
        newton_max_iter=solver_cfg.get('newton_max_iter', 5),
    )


def build_neuron(cfg, graph):
    n = cfg.get('neuron', {})
    if not n.get('enabled', False):
        return None

    return NeuronLayer(
        graph=graph,
        neuron_nodes=n.get('neuron_nodes', []),
        dt=n.get('dt', 0.01),
        use_nernst=n.get('use_nernst', False),
    )


def build_synapses(cfg, graph, dt=0.01):
    s = cfg.get('synapse', {})
    if not s.get('enabled', False):
        return None

    synapse_list = s.get('synapses', [])
    return SynapseLayer(
        graph=graph,
        synapses=synapse_list,
        dt=dt,
        V_threshold=s.get('V_threshold', -20.0),
        use_stdp=s.get('use_stdp', False),
        A_plus=s.get('A_plus', 0.01),
        A_minus=s.get('A_minus', 0.012),
        tau_plus=s.get('tau_plus', 20.0),
        tau_minus=s.get('tau_minus', 20.0),
        w_min=s.get('w_min', 0.0),
        w_max=s.get('w_max', 10.0),
    )


def run_simulation(cfg, n_steps=None, output_every=None):
    sim_cfg = cfg.get('simulation', {})
    n_steps = n_steps if n_steps is not None else sim_cfg.get('n_steps', 100)
    output_every = (output_every if output_every is not None
                    else sim_cfg.get('output_every', 10))

    graph = build_graph(cfg)
    ions = build_ions(cfg)
    solver = build_solver(cfg, graph, ions)
    neuron = build_neuron(cfg, graph)
    synapse = build_synapses(cfg, graph, dt=solver.dt)

    print("Bio-COMSOL симуляция")
    print(f"  Граф: {graph.N} узлов, {len(graph.edges)} рёбер")
    print(f"  Ионов: {solver.n_ions} — {solver.names}")
    print(f"  Нейрон: {'вкл' if neuron else 'выкл'}")
    if neuron:
        print(f"    Узлы: {list(neuron.neuron_indices)}")
        print(f"    Нернст: {'да' if neuron.use_nernst else 'нет'}")
    print(f"  Синапсы: {'вкл' if synapse else 'выкл'}")
    if synapse:
        print(f"    Синапсов: {synapse.n_syn}")
        print(f"    STDP: {'да' if synapse.use_stdp else 'нет'}")
    print(f"  Самосогласование: {'да' if solver.self_consistent else 'нет'}")
    print(f"  Метод: {'Ньютон' if solver.use_newton else 'Пикар'}")
    print(f"  Шагов: {n_steps}")
    print(f"  dt: {solver.dt}")
    print()

    for step in range(n_steps):
        iters = solver.step()

        if neuron is not None:
            if synapse is not None:
                synapse.step(solver.phi)
                I_ext = synapse.get_current_array(
                    solver.phi, neuron.neuron_indices)
            else:
                I_ext = None
            solver.phi = neuron.apply_to_graph(
                solver.phi, solver.c, I_ext=I_ext)

        if (step + 1) % output_every == 0:
            qn_err = solver.quasineutrality_error()
            charge = solver.total_charge()
            method = "N" if solver.use_newton else "P"
            print(f"  Шаг {step + 1:5d} | {method}: {iters} | "
                  f"квазинейтральность: {qn_err:.2e} | "
                  f"заряд: {charge:.2e}")

    print()
    print("Симуляция завершена.")
    return solver, neuron


def main():
    parser = argparse.ArgumentParser(description='Bio-COMSOL симуляция')
    parser.add_argument('config', help='Путь к config.yaml')
    parser.add_argument('--steps', type=int, default=None,
                        help='Число шагов (по умолчанию из конфига)')
    parser.add_argument('--output-every', type=int, default=None,
                        help='Вывод каждые N шагов (по умолчанию из конфига)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_simulation(cfg, args.steps, args.output_every)


if __name__ == '__main__':
    main()
# === END ===
