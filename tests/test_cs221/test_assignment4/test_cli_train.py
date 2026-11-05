import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import dnnlpy.cs221.assignment4 as a4
from dnnlpy.cli.cs221.assignment4 import train


@pytest.mark.parametrize(
    ('agent', 'filename'),
    [
        ('value-iteration', 'vi_trial_1_weights.json'),
        ('tabular', 'tabular_trial_1_weights.npy'),
        ('function-approx', 'function_approx_trial_1_weights.npy'),
        ('constrained', 'constrained_trial_1_weights.npy'),
    ],
)
def test_train_cli_saves_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent: str, filename: str
):
    def fake_simulate(mdp: Any, rl: Any, *, num_episodes: int, **kwargs: Any) -> Any:
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
            '--num_trials',
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
        assert weights.ndim == 2 and weights.shape[0] > 0
        assert weights.shape[1] == 3
        assert np.issubdtype(weights.dtype, np.floating)
        assert np.all(np.isfinite(weights))
