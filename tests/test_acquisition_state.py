from __future__ import annotations

import json

import pytest

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound


@pytest.mark.parametrize(
    ("acquisition_type", "parameter"),
    [(UpperConfidenceBound, "kappa"), (ProbabilityOfImprovement, "xi"), (ExpectedImprovement, "xi")],
)
@pytest.mark.parametrize(
    ("delay", "suggestions_before_save"),
    [(3, 2), (3, 4), (None, 2), (0, 2), (3, 0)],
    ids=["before-delay", "after-delay", "no-delay", "zero-delay", "unused-acquisition"],
)
def test_save_load_preserves_exploration_schedule(
    tmp_path, acquisition_type, parameter, delay, suggestions_before_save
):
    def make_optimizer():
        acquisition = acquisition_type(
            **{parameter: 1.0}, exploration_decay=0.5, exploration_decay_delay=delay
        )
        optimizer = BayesianOptimization(
            f=lambda x: -((x - 0.3) ** 2),
            pbounds={"x": (-1.0, 1.0)},
            acquisition_function=acquisition,
            random_state=7,
            verbose=0,
        )
        optimizer.set_gp_params(n_restarts_optimizer=0)
        return optimizer

    original = make_optimizer()
    for x in (-1.0, 0.0, 1.0):
        original.probe({"x": x}, lazy=False)
    for _ in range(suggestions_before_save):
        original.suggest()

    state_path = tmp_path / "optimizer.json"
    original.save_state(state_path)
    restored = make_optimizer()
    restored.load_state(state_path)

    assert restored.res == original.res
    assert getattr(restored.acquisition_function, parameter) == getattr(
        original.acquisition_function, parameter
    )

    for _ in range(2):
        assert restored.suggest() == pytest.approx(original.suggest())
        assert getattr(restored.acquisition_function, parameter) == getattr(
            original.acquisition_function, parameter
        )


@pytest.mark.parametrize(
    ("acquisition_type", "parameter"),
    [(UpperConfidenceBound, "kappa"), (ProbabilityOfImprovement, "xi"), (ExpectedImprovement, "xi")],
)
def test_load_legacy_acquisition_state(tmp_path, acquisition_type, parameter):
    acquisition = acquisition_type(**{parameter: 1.0}, exploration_decay=0.5, exploration_decay_delay=3)
    original = BayesianOptimization(
        f=lambda x: x, pbounds={"x": (-1.0, 1.0)}, acquisition_function=acquisition, random_state=7, verbose=0
    )
    state_path = tmp_path / "legacy.json"
    original.save_state(state_path)
    state = json.loads(state_path.read_text())
    state["acquisition_params"].pop("i", None)
    state_path.write_text(json.dumps(state))

    restored_acquisition = acquisition_type(**{parameter: 2.0})
    restored = BayesianOptimization(
        f=lambda x: x,
        pbounds={"x": (-1.0, 1.0)},
        acquisition_function=restored_acquisition,
        random_state=7,
        verbose=0,
    )
    restored.load_state(state_path)
    assert getattr(restored_acquisition, parameter) == 1.0
    assert restored_acquisition.exploration_decay == 0.5
    assert restored_acquisition.exploration_decay_delay == 3
    assert restored.suggest() == original.suggest()
