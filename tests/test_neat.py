import json
import random
import sys
from pathlib import Path

import neat
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import train  # noqa: E402


def fresh_population(pop_size=30, seed=0):
    random.seed(seed)
    config = common.load_config()
    config.pop_size = pop_size
    p = neat.Population(config)
    train.unify_initial_hidden_keys(p, config)
    return p, config


def mutated_genome(config, n=30, seed=1):
    random.seed(seed)
    g = neat.DefaultGenome(1)
    g.configure_new(config.genome_config)
    for _ in range(n):
        g.mutate(config.genome_config)
    return g


def test_initial_genome_has_hidden():
    p, config = fresh_population()
    gc = config.genome_config
    inputs, outputs = set(gc.input_keys), set(gc.output_keys)
    hidden_sets = set()
    for g in p.population.values():
        hidden = frozenset(k for k in g.nodes if k not in outputs)
        hidden_sets.add(hidden)
        assert len(hidden) == gc.num_hidden == 8
        assert not any(a in inputs and b in outputs for a, b in g.connections)  # full_nodirect
        assert common.topology(g, config)["hidden_in_use"] == 8
    # All genomes share the same hidden-node ids, so they are not one species each.
    assert len(hidden_sets) == 1
    assert len(p.species.species) < len(p.population) // 2


def test_hof_rejects_linear():
    config = common.load_config()
    g = mutated_genome(config, n=5)
    assert common.has_hidden_layer(g, config)
    gc = config.genome_config
    # Strip all hidden nodes and wire inputs straight to outputs: a linear net.
    g.nodes = {k: v for k, v in g.nodes.items() if k in gc.output_keys}
    g.connections = {}
    for i in gc.input_keys:
        for o in gc.output_keys:
            cg = g.create_connection(gc, i, o)
            g.connections[cg.key] = cg
    assert common.topology(g, config)["hidden_in_use"] == 0
    assert not common.has_hidden_layer(g, config)
    assert train.linear_penalty(100.0) == 50.0
    assert train.linear_penalty(-20.0) == -70.0


def test_dangling_hidden_not_counted():
    config = common.load_config()
    gc = config.genome_config
    g = neat.DefaultGenome(1)
    g.configure_new(config.genome_config)
    # Disable every hidden->output link: hidden nodes exist but nothing uses them.
    for (a, b), cg in g.connections.items():
        if b in gc.output_keys:
            cg.enabled = False
    assert common.topology(g, config)["hidden_in_use"] == 0


def test_fastnet_matches_neat():
    config = common.load_config()
    rng = np.random.default_rng(0)
    for seed in range(5):
        g = mutated_genome(config, n=30, seed=seed)
        ref = neat.nn.FeedForwardNetwork.create(g, config)
        fast = common.FastNet(g, config)
        for x in rng.uniform(-1, 1, (50, 24)):
            np.testing.assert_allclose(fast.activate(x), ref.activate(x), atol=1e-9)


def test_deterministic_episode():
    config = common.load_config()
    g = mutated_genome(config, n=10)
    env = common.make_env()
    r1 = common.run_episode(common.make_net(g, config), env, 123)
    r2 = common.run_episode(common.make_net(g, config), common.make_env(), 123)
    assert r1 == r2


@pytest.mark.smoke
def test_smoke_train_resume_applies_new_config(tmp_path):
    runs = tmp_path / "runs"
    base = ["--name", "smoke", "--runs-dir", str(runs), "--workers", "2", "--k", "1",
            "--hof-every", "1", "--hof-top", "2", "--hof-seeds", "2", "--checkpoint-every", "1"]
    p, state = train.main(base + ["--pop-size", "10", "--generations", "2"])
    run = runs / "smoke"
    assert p.generation == 2
    assert (run / "stats.csv").exists()
    assert (run / "checkpoints" / "latest.pkl.gz").exists()
    assert (run / "best" / "genome.pkl").exists()
    genome, config, meta = common.resolve_genome(run)
    assert common.has_hidden_layer(genome, config)

    # Change pop_size and a mutation rate in a NEW ini, resume: the new values must be used.
    ini = (common.DEFAULT_CONFIG).read_text()
    ini = ini.replace("pop_size              = 200", "pop_size              = 14")
    ini = ini.replace("node_add_prob         = 0.1", "node_add_prob         = 0.42")
    new_ini = tmp_path / "changed.ini"
    new_ini.write_text(ini)
    p2, _ = train.main(base + ["--config", str(new_ini), "--resume", "--generations", "3"])
    assert p2.generation == 3  # continued from 2, not 0
    assert p2.config.pop_size == 14
    assert p2.config.genome_config.node_add_prob == 0.42
    eff = json.loads((run / "run_config.json").read_text())["effective"]
    assert eff["pop_size"] == 14 and eff["node_add_prob"] == 0.42
    # Reproduction of the restored population used the new pop_size.
    assert abs(len(p2.population) - 14) <= 2


@pytest.mark.smoke
def test_fall_penalty_and_hof_rescore(tmp_path):
    runs = tmp_path / "runs"
    base = ["--name", "fp", "--runs-dir", str(runs), "--workers", "2", "--k", "1",
            "--hof-every", "1", "--hof-top", "1", "--hof-seeds", "2", "--checkpoint-every", "1"]
    train.main(base + ["--pop-size", "8", "--generations", "1"])
    (runs / "fp" / "overrides.json").write_text('{"fall_penalty": 100, "hof_seeds": 3}')
    p, state = train.main(base + ["--resume", "--generations", "2"])
    eff = json.loads((runs / "fp" / "run_config.json").read_text())["effective"]
    assert eff["fall_penalty"] == 100 and eff["hof_seeds"] == 3
    assert state["hof_seeds"] == 3
    _, meta = common.load_genome(runs / "fp" / "best" / "genome.pkl")
    assert len(meta["held_out_seeds"]) == 3   # champion re-scored on the new terrain set
