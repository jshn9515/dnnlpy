import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

import dnnlpy.cs221.assignment4 as a4
from dnnlpy.cli.cs221.assignment4 import eval, train


def _disable_eval_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    original_env = a4.CustomMountainCarEnv

    def headless_env(*args: Any, **kwargs: Any) -> a4.CustomMountainCarEnv:
        assert kwargs.pop('render_mode') == 'human'
        return original_env(*args, **kwargs)

    monkeypatch.setattr(a4, 'CustomMountainCarEnv', headless_env)


@pytest.mark.parametrize(
    ('agent', 'agent_class'),
    [
        ('value-iteration', 'ModelBasedMonteCarlo'),
        ('tabular', 'TabularQLearning'),
        ('function-approx', 'FunctionApproxQLearning'),
        ('constrained', 'ConstrainedQLearning'),
        ('policy-gradient', 'PolicyGradient'),
        ('actor-critic', 'ActorCritic'),
    ],
)
def test_train_cli_uses_distinct_trial_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
    agent_class: str,
):
    monkeypatch.setattr(
        a4,
        'simulate',
        lambda mdp, rl, num_episodes, **kwargs: np.zeros(num_episodes),
    )
    monkeypatch.setattr(train, 'plot_rewards', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-train',
            '--agent',
            agent,
            '--num-trials',
            '2',
            '--num-train-episodes',
            '1',
            '--num-eval-episodes',
            '1',
            '--seed',
            '11',
            '--path',
            str(tmp_path),
        ],
    )

    with (
        patch.object(a4, 'CustomMountainCarEnv', wraps=a4.CustomMountainCarEnv) as env,
        patch.object(a4, agent_class, wraps=getattr(a4, agent_class)) as rl,
    ):
        train.main()

    assert [call.kwargs['seed'] for call in env.call_args_list] == [11, 12]
    assert [call.kwargs['seed'] for call in rl.call_args_list] == [11, 12]


def test_train_cli_all_agents_share_seed_per_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        a4,
        'simulate',
        lambda mdp, rl, num_episodes, **kwargs: np.zeros(num_episodes),
    )
    monkeypatch.setattr(train, 'plot_training_curves', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'dnnlpy-cs221-a4-train',
            '--agent',
            'all',
            '--num-trials',
            '2',
            '--num-train-episodes',
            '1',
            '--seed',
            '11',
            '--path',
            str(tmp_path),
        ],
    )

    agent_classes = [
        'ModelBasedMonteCarlo',
        'TabularQLearning',
        'FunctionApproxQLearning',
        'ConstrainedQLearning',
        'PolicyGradient',
        'ActorCritic',
    ]
    with ExitStack() as stack:
        env = stack.enter_context(
            patch.object(a4, 'CustomMountainCarEnv', wraps=a4.CustomMountainCarEnv)
        )
        agents = [
            stack.enter_context(patch.object(a4, name, wraps=getattr(a4, name)))
            for name in agent_classes
        ]
        train.main()

    seed1 = [call.kwargs['seed'] for call in env.call_args_list]
    seed2 = [call.kwargs['seed'] for agent in agents for call in agent.call_args_list]
    assert seed1 == [11] * 6 + [12] * 6
    assert seed2 == [11, 12] * 6


@pytest.mark.parametrize(
    ('agent', 'filename', 'options'),
    [
        ('value-iteration', 'value_iteration_0.json', []),
        ('tabular', 'tabular_0.npy', []),
        ('function-approx', 'function_approx_0_fourier_norm.npy', []),
        ('constrained', 'constrained_0_fourier_norm.npy', []),
        ('policy-gradient', 'policy_gradient_0_fourier_norm.npy', []),
        ('actor-critic', 'actor_critic_0_fourier_norm.npy', []),
        ('function-approx', 'function_approx_0_fourier_raw.npy', ['--no-normalized']),
        ('constrained', 'constrained_0_fourier_raw.npy', ['--no-normalized']),
        ('policy-gradient', 'policy_gradient_0_fourier_raw.npy', ['--no-normalized']),
        ('actor-critic', 'actor_critic_0_fourier_raw.npy', ['--no-normalized']),
        (
            'policy-gradient',
            'policy_gradient_0_fourier_norm_nobase.npy',
            ['--no-baseline'],
        ),
        ('actor-critic', 'actor_critic_0_fourier_norm_nobase.npy', ['--no-baseline']),
        (
            'actor-critic',
            'actor_critic_0_fourier_raw_nobase.npy',
            ['--no-normalized', '--no-baseline'],
        ),
        (
            'function-approx',
            'function_approx_0_poly_norm.npy',
            ['--feature-extractor', 'poly'],
        ),
        (
            'constrained',
            'constrained_0_poly_raw.npy',
            ['--feature-extractor', 'poly', '--no-normalized'],
        ),
        (
            'policy-gradient',
            'policy_gradient_0_poly_norm.npy',
            ['--feature-extractor', 'poly'],
        ),
        (
            'actor-critic',
            'actor_critic_0_poly_raw_nobase.npy',
            ['--feature-extractor', 'poly', '--no-normalized', '--no-baseline'],
        ),
    ],
)
def test_train_cli_saves_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
    filename: str,
    options: list[str],
):
    def fake_simulate(mdp: Any, rl: Any, num_episodes: int, **kwargs: Any) -> Any:
        if agent == 'value-iteration':
            rl.pi_indices[0] = 1

        elif agent != 'tabular':
            state = np.array([-0.5, 0.02])
            degree = 3 if agent in {'policy-gradient', 'actor-critic'} else 5
            low = mdp.base_env.low if '--no-normalized' not in options else None
            high = mdp.base_env.high if '--no-normalized' not in options else None

            if '--feature-extractor' in options:
                expected = a4.polynomial_feature_extractor(
                    state, degree, low=low, high=high
                )
            else:
                expected = a4.fourier_feature_extractor(
                    state, degree, low=low, high=high
                )

            actual = rl.feature_extractor(state)
            npt.assert_allclose(actual, expected)

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
            *options,
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
    ('agent', 'filename', 'attribute', 'options'),
    [
        ('value-iteration', 'value_iteration_0.json', 'pi_indices', []),
        ('tabular', 'tabular_0.npy', 'Q', []),
        ('function-approx', 'function_approx_0_fourier_norm.npy', 'weight', []),
        ('constrained', 'constrained_0_fourier_norm.npy', 'weight', []),
        ('policy-gradient', 'policy_gradient_0_fourier_norm.npy', 'theta', []),
        ('actor-critic', 'actor_critic_0_fourier_norm.npy', 'theta', []),
        (
            'function-approx',
            'function_approx_0_fourier_raw.npy',
            'weight',
            ['--no-normalized'],
        ),
        ('constrained', 'constrained_0_fourier_raw.npy', 'weight', ['--no-normalized']),
        (
            'policy-gradient',
            'policy_gradient_0_fourier_raw.npy',
            'theta',
            ['--no-normalized'],
        ),
        (
            'actor-critic',
            'actor_critic_0_fourier_raw.npy',
            'theta',
            ['--no-normalized'],
        ),
        (
            'policy-gradient',
            'policy_gradient_0_fourier_norm_nobase.npy',
            'theta',
            ['--no-baseline'],
        ),
        (
            'actor-critic',
            'actor_critic_0_fourier_norm_nobase.npy',
            'theta',
            ['--no-baseline'],
        ),
        (
            'actor-critic',
            'actor_critic_0_fourier_raw_nobase.npy',
            'theta',
            ['--no-normalized', '--no-baseline'],
        ),
        (
            'function-approx',
            'function_approx_0_poly_norm.npy',
            'weight',
            ['--feature-extractor', 'poly'],
        ),
        (
            'constrained',
            'constrained_0_poly_raw.npy',
            'weight',
            ['--feature-extractor', 'poly', '--no-normalized'],
        ),
        (
            'policy-gradient',
            'policy_gradient_0_poly_norm.npy',
            'theta',
            ['--feature-extractor', 'poly'],
        ),
        (
            'actor-critic',
            'actor_critic_0_poly_raw_nobase.npy',
            'theta',
            ['--feature-extractor', 'poly', '--no-normalized', '--no-baseline'],
        ),
    ],
)
def test_eval_cli_loads_trained_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
    filename: str,
    attribute: str,
    options: list[str],
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
            *options,
        ],
    )
    train.main()
    assert (tmp_path / filename).is_file()

    calls.clear()
    _disable_eval_rendering(monkeypatch)
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
            *options,
        ],
    )
    eval.main()

    assert len(calls) == 1
    _, rl, num_episodes, kwargs = calls[0]
    assert num_episodes == 1

    expected_kwargs: dict[str, Any] = {
        'train': False,
        'verbose': False,
        'demo': True,
    }
    if agent in {'policy-gradient', 'actor-critic'}:
        expected_kwargs['eval_mode'] = 'sample'

    assert kwargs == expected_kwargs

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
    _disable_eval_rendering(monkeypatch)
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
    eval.main()

    assert len(calls) == 1
    rl, num_episodes, kwargs = calls[0]
    assert isinstance(rl, a4.RandomRLAlgorithm)
    assert num_episodes == 1
    assert kwargs == {'train': False, 'verbose': False, 'demo': True}


@pytest.mark.parametrize('plot_format', ['png', 'svg', 'pdf'])
def test_train_cli_all_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plot_format: str
):
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
            '--format',
            plot_format,
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
    assert path == tmp_path / f'training_curves_trial_0.{plot_format}'
    assert show is False
