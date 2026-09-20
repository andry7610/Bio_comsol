"""tests/test_main.py — Тесты конфиг-драйвенного запуска."""

import numpy as np
import pytest
import yaml

from main import (
    load_config, build_graph, build_ions,
    build_solver, build_neuron, run_simulation,
)


@pytest.fixture
def cfg():
    return {
        'graph': {
            'N': 20, 'topology': 'chain',
            'conductivity': 1.0, 'cross_section': 1e-8,
        },
        'physics_constants': {
            'T': 310.0, 'R': 8.314, 'F': 96485.0,
        },
        'ion_species': {
            'Na': {'D': 1.33e-9, 'z': 1, 'c0': 145.0},
            'K': {'D': 1.96e-9, 'z': 1, 'c0': 4.0},
            'Cl': {'D': 2.03e-9, 'z': -1, 'c0': 110.0},
            'Ca': {'D': 0.79e-9, 'z': 2, 'c0': 1.0},
        },
        'solver': {
            'dt': 0.01, 'membrane_potential': -1.8,
            'picard_tol': 1e-6, 'picard_max_iter': 20,
        },
        'neuron': {
            'enabled': True, 'neuron_nodes': [5, 10, 15],
            'dt': 0.01, 'use_nernst': False,
        },
        'simulation': {'n_steps': 10, 'output_every': 5},
    }


@pytest.fixture
def cfg_file(tmp_path, cfg):
    path = tmp_path / "test_config.yaml"
    with open(path, 'w') as f:
        yaml.dump(cfg, f)
    return str(path)


class TestLoadConfig:
    def test_load_yaml(self, cfg_file):
        c = load_config(cfg_file)
        assert 'graph' in c
        assert 'ion_species' in c
        assert 'solver' in c

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")


class TestBuildGraph:
    def test_graph_from_config(self, cfg):
        g = build_graph(cfg)
        assert g.N == 20
        assert len(g.edges) > 0

    def test_graph_defaults(self):
        g = build_graph({})
        assert g.N == 20

    def test_graph_ring(self, cfg):
        cfg['graph']['topology'] = 'ring'
        g = build_graph(cfg)
        assert g.N == 20


class TestBuildIons:
    def test_ions_count(self, cfg):
        ions = build_ions(cfg)
        assert len(ions) == 4

    def test_ion_names(self, cfg):
        ions = build_ions(cfg)
        names = [i['name'] for i in ions]
        assert 'Na' in names
        assert 'Ca' in names

    def test_ion_defaults(self):
        c = {'ion_species': {'Na': {'D': 1e-9, 'z': 1, 'c0': 100}}}
        ions = build_ions(c)
        assert ions[0]['c_left'] == 100
        assert ions[0]['c_right'] == 100

    def test_ion_custom_boundaries(self):
        c = {'ion_species': {
            'Na': {'D': 1e-9, 'z': 1, 'c0': 100, 'c_left': 145, 'c_right': 10}
        }}
        ions = build_ions(c)
        assert ions[0]['c_left'] == 145
        assert ions[0]['c_right'] == 10


class TestBuildSolver:
    def test_solver_from_config(self, cfg):
        g = build_graph(cfg)
        ions = build_ions(cfg)
        s = build_solver(cfg, g, ions)
        assert s.N == 20
        assert s.n_ions == 4
        assert s.dt == 0.01

    def test_solver_F_RT(self, cfg):
        g = build_graph(cfg)
        ions = build_ions(cfg)
        s = build_solver(cfg, g, ions)
        assert abs(s.F_RT - 38.94) < 0.1

    def test_solver_defaults(self):
        g = build_graph({})
        ions = build_ions({})
        s = build_solver({}, g, ions)
        assert s.n_ions == 0


class TestBuildNeuron:
    def test_neuron_enabled(self, cfg):
        g = build_graph(cfg)
        n = build_neuron(cfg, g)
        assert n is not None
        assert len(n.neuron_indices) == 3

    def test_neuron_disabled(self, cfg):
        cfg['neuron']['enabled'] = False
        g = build_graph(cfg)
        n = build_neuron(cfg, g)
        assert n is None

    def test_neuron_missing_section(self):
        g = build_graph({})
        n = build_neuron({}, g)
        assert n is None


class TestRunSimulation:
    def test_run_basic(self, cfg):
        solver, neuron = run_simulation(cfg, n_steps=10, output_every=5)
        assert solver.step_count == 10
        assert neuron is not None

    def test_run_no_neuron(self, cfg):
        cfg['neuron']['enabled'] = False
        solver, neuron = run_simulation(cfg, n_steps=5, output_every=5)
        assert solver.step_count == 5
        assert neuron is None

    def test_run_no_nan(self, cfg):
        solver, neuron = run_simulation(cfg, n_steps=50, output_every=50)
        assert not np.any(np.isnan(solver.c))
        assert not np.any(np.isnan(solver.phi))

    def test_run_from_sim_section(self, cfg):
        """n_steps и output_every берутся из cfg['simulation']."""
        solver, neuron = run_simulation(cfg)
        assert solver.step_count == 10

    def test_run_cli_override(self, cfg):
        """--steps перекрывает cfg['simulation']['n_steps']."""
        solver, neuron = run_simulation(cfg, n_steps=25, output_every=25)
        assert solver.step_count == 25

    def test_run_charge_not_exploded(self, cfg):
        solver, neuron = run_simulation(cfg, n_steps=100, output_every=100)
        charge = solver.total_charge()
        assert abs(charge) < 1e6
# === END ===
