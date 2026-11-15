import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4
from dnnlpy.cli.cs221.assignment4 import evaluate, train


@pytest.mark.parametrize(
    ('agent', 'filename'),
    [
        ('value-iteration', 'vi_trial_0_weights.json'),
        ('tabular', 'tabular_trial_0_weights.npy'),
        ('function-approx', 'function_approx_trial_0_weights.npy'),
        ('constrained', 'constrained_trial_0_weights.npy'),
        ('policy-gradient', 'policy_gradient_trial_0_weights.npy'),
        ('actor-critic', 'actor_critic_trial_0_weights.npy'),
    ],
)
def test_train_cli_saves_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent: str, filename: str
):
    def fake_simulate(mdp: Any, rl: Any, num_episodes: int, **kwargs: Any) -> Any:
        if agent == 'value-iteration':
            rl.pi_indices[0] = 1
        return np.zeros(num_episodes)

    monkeypatch.setattr(a4, 'simulate', fake_simulate)
    monkeypatch.setattr(train, 'plot_rewards', lambda *args, **kwargs: None)

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-train',
            '--agent',
            agent,
            '--num-trials',
            '1',
            '--path',
            str(tmp_path),
        ],
    )

    train.main()

    output = tmp_path / filename
    assert output.is_file()

    if agent == 'value-iteration':
        policy = json.loads(output.read_text())
        assert isinstance(policy, dict) and policy
        for key, value in policy.items():
            assert key.isdecimal()
            assert value in {0, 1, 2}
    else:
        weights = np.load(output, allow_pickle=False)
        assert weights.ndim == 2
        assert weights.shape[0] == (
            16
            if agent in {'policy-gradient', 'actor-critic'}
            else 36
            if agent in {'function-approx', 'constrained'}
            else 400
        )
        assert weights.shape[1] == 3
        assert np.issubdtype(weights.dtype, np.floating)
        assert np.all(np.isfinite(weights))


@pytest.mark.parametrize(
    ('agent', 'filename', 'attribute'),
    [
        ('value-iteration', 'vi_trial_0_weights.json', 'pi_indices'),
        ('tabular', 'tabular_trial_0_weights.npy', 'Q'),
        ('function-approx', 'function_approx_trial_0_weights.npy', 'weight'),
        ('constrained', 'constrained_trial_0_weights.npy', 'weight'),
        ('policy-gradient', 'policy_gradient_trial_0_weights.npy', 'theta'),
        ('actor-critic', 'actor_critic_trial_0_weights.npy', 'theta'),
    ],
)
def test_eval_cli_loads_trained_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
    filename: str,
    attribute: str,
):
    calls: list[tuple[Any, Any, int, dict[str, Any]]] = []

    def fake_simulate(
        mdp: Any, rl: Any, num_episodes: int, **kwargs: Any
    ) -> np.ndarray:
        calls.append((mdp, rl, num_episodes, kwargs))

        if len(calls) == 1:
            if agent == 'value-iteration':
                rl.pi_indices[0] = 1
            else:
                getattr(rl, attribute)[0, 0] = 2.5

        return np.zeros(num_episodes)

    monkeypatch.setattr(a4, 'simulate', fake_simulate)
    monkeypatch.setattr(train, 'plot_rewards', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-train',
            '--agent',
            agent,
            '--num-trials',
            '1',
            '--path',
            str(tmp_path),
        ],
    )
    train.main()
    assert (tmp_path / filename).is_file()

    calls.clear()
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-eval',
            '--agent',
            agent,
            '--trial',
            '0',
            '--path',
            str(tmp_path),
        ],
    )
    evaluate.main()

    assert len(calls) == 1
    _, rl, num_episodes, kwargs = calls[0]
    assert num_episodes == 1
    assert kwargs == {'train': False, 'verbose': False, 'demo': True}

    if agent == 'value-iteration':
        policy = json.loads((tmp_path / filename).read_text())
        for state, action in policy.items():
            assert rl.actions[rl.pi_indices[int(state)]] == action
    else:
        npt.assert_equal(
            getattr(rl, attribute),
            np.load(tmp_path / filename, allow_pickle=False),
        )
        assert getattr(rl, attribute)[0, 0] == 2.5


def test_eval_cli_naive_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[tuple[Any, int, dict[str, Any]]] = []

    def fake_simulate(
        mdp: Any, rl: Any, num_episodes: int, **kwargs: Any
    ) -> np.ndarray:
        calls.append((rl, num_episodes, kwargs))
        return np.zeros(num_episodes)

    monkeypatch.setattr(a4, 'simulate', fake_simulate)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-eval',
            '--agent',
            'naive',
            '--path',
            str(tmp_path),
        ],
    )
    evaluate.main()

    assert len(calls) == 1
    rl, num_episodes, kwargs = calls[0]
    assert isinstance(rl, a4.RandomRLAlgorithm)
    assert num_episodes == 1
    assert kwargs == {'train': False, 'verbose': False, 'demo': True}


def test_train_cli_all_agents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    agents: list[type[Any]] = []
    plotted: list[tuple[list[tuple[str, np.ndarray]], Path, bool]] = []

    def fake_simulate(
        mdp: Any, rl: Any, num_episodes: int, **kwargs: Any
    ) -> np.ndarray:
        agents.append(type(rl))
        assert num_episodes == 2
        assert kwargs == {}
        return np.zeros(num_episodes)

    def fake_plot(
        curves: list[tuple[str, np.ndarray]],
        path: Path,
        show: bool,
    ):
        plotted.append((curves, path, show))

    monkeypatch.setattr(a4, 'simulate', fake_simulate)
    monkeypatch.setattr(train, 'plot_training_curves', fake_plot)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-train',
            '--agent',
            'all',
            '--num-trials',
            '1',
            '--num-train-episodes',
            '2',
            '--no-show-plot',
            '--path',
            str(tmp_path),
        ],
    )
    train.main()

    assert agents == [
        a4.ModelBasedMonteCarlo,
        a4.TabularQLearning,
        a4.FunctionApproxQLearning,
        a4.ConstrainedQLearning,
        a4.PolicyGradient,
        a4.ActorCritic,
    ]
    assert len(plotted) == 1
    curves, path, show = plotted[0]
    assert [name for name, _ in curves] == [
        'value-iteration',
        'tabular',
        'function-approx',
        'constrained',
        'policy-gradient',
        'actor-critic',
    ]
    assert all(np.array_equal(rewards, np.zeros(2)) for _, rewards in curves)
    assert path == tmp_path / 'training_curves_trial_0.svg'
    assert show is False
