from typing import Final, Literal

import pytest

import dnnlpy.cs234.assignment1 as a1

VAL_FUNC_LEFT: Final[float] = 30.328
VAL_FUNC_RIGHT: Final[float] = 36.859


class TestRiverSwim:
    seed: int = 1234
    current: Literal['weak'] = 'weak'
    gamma: float = 0.99
    tolerance: float = 1e-3

    def setup_method(self):
        self.env = a1.RiverSwimEnv(self.current, self.seed)
        self.rl = a1.RiverSwimAgent(self.env, self.gamma, self.tolerance)

    def test_rl_agent_rejects_invalid_gamma(self):
        with pytest.raises(AssertionError):
            a1.RiverSwimAgent(self.env, gamma=1.5, tol=self.tolerance)

    def test_policy_iteration(self):
        result = self.rl.policy_iteration()

        assert isinstance(result, a1.RiverSwimResult)
        assert result.val_func[0] == pytest.approx(VAL_FUNC_LEFT, abs=1e-3, rel=0)
        assert result.val_func[-1] == pytest.approx(VAL_FUNC_RIGHT, abs=1e-3, rel=0)

    def test_value_iteration(self):
        result = self.rl.value_iteration()

        assert isinstance(result, a1.RiverSwimResult)
        assert result.val_func[0] == pytest.approx(VAL_FUNC_LEFT, abs=1e-3, rel=0)
        assert result.val_func[-1] == pytest.approx(VAL_FUNC_RIGHT, abs=1e-3, rel=0)
