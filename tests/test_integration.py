from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint
import json

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import (
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    ExpectedImprovement,
    ConstantLiar,
    GPHedge
)

# Test fixtures
@pytest.fixture
def target_func():
    return lambda x, y: -(x - 1)**2 - (y - 2)**2  # Maximum at (1,2)

@pytest.fixture
def pbounds():
    return {"x": (-5, 5), "y": (-5, 5)}

@pytest.fixture
def constraint_func():
    return lambda x, y: x + y  # Simple constraint: sum of parameters

@pytest.fixture
def constraint(constraint_func):
    return NonlinearConstraint(
        fun=constraint_func,
        lb=-1.0,
        ub=4.0
    )

def verify_optimizers_match(optimizer1, optimizer2):
    """Helper function to verify two optimizers match."""
    assert len(optimizer1.space) == len(optimizer2.space)
    assert optimizer1.max["target"] == optimizer2.max["target"]
    assert optimizer1.max["params"] == optimizer2.max["params"]
    
    
    
    np.testing.assert_array_equal(optimizer1.space.params, optimizer2.space.params)
    np.testing.assert_array_equal(optimizer1.space.target, optimizer2.space.target)
    
    if optimizer1.is_constrained:
        np.testing.assert_array_equal(
            optimizer1.space._constraint_values,
            optimizer2.space._constraint_values
        )
        assert optimizer1.space._constraint.lb == optimizer2.space._constraint.lb
        assert optimizer1.space._constraint.ub == optimizer2.space._constraint.ub
    
    assert np.random.get_state()[1][0] == np.random.get_state()[1][0]
    
    assert optimizer1._gp.kernel.get_params() == optimizer2._gp.kernel.get_params()
    
    suggestion1 = optimizer1.suggest()
    suggestion2 = optimizer2.suggest()
    assert suggestion1 == suggestion2, f"\nSuggestion 1: {suggestion1}\nSuggestion 2: {suggestion2}"
    

    

def test_integration_upper_confidence_bound(target_func, pbounds, tmp_path):
    """Test save/load integration with UpperConfidenceBound acquisition."""
    acquisition_function = UpperConfidenceBound(kappa=2.576)
    
    # Create and run first optimizer
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=2, n_iter=3)
    
    # Save state
    state_path = tmp_path / "ucb_state.json"
    optimizer.save_state(state_path)
    
    # Create new optimizer and load state
    new_optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=UpperConfidenceBound(kappa=2.576),
        random_state=1,
        verbose=0
    )
    new_optimizer.load_state(state_path)
    
    verify_optimizers_match(optimizer, new_optimizer)

def test_integration_probability_improvement(target_func, pbounds, tmp_path):
    """Test save/load integration with ProbabilityOfImprovement acquisition."""
    acquisition_function = ProbabilityOfImprovement(xi=0.01)
    
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=2, n_iter=3)
    
    state_path = tmp_path / "pi_state.json"
    optimizer.save_state(state_path)
    
    new_optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=ProbabilityOfImprovement(xi=0.01),
        random_state=1,
        verbose=0
    )
    new_optimizer.load_state(state_path)
    
    verify_optimizers_match(optimizer, new_optimizer)

def test_integration_expected_improvement(target_func, pbounds, tmp_path):
    """Test save/load integration with ExpectedImprovement acquisition."""
    acquisition_function = ExpectedImprovement(xi=0.01)
    
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=2, n_iter=3)
    
    state_path = tmp_path / "ei_state.json"
    optimizer.save_state(state_path)
    
    new_optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        random_state=1,
        verbose=0
    )
    new_optimizer.load_state(state_path)
    
    verify_optimizers_match(optimizer, new_optimizer)

def test_integration_constant_liar(target_func, pbounds, tmp_path):
    """Test save/load integration with ConstantLiar acquisition."""
    base_acq = UpperConfidenceBound(kappa=2.576)
    acquisition_function = ConstantLiar(base_acquisition=base_acq)
    
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=2, n_iter=3)
    
    state_path = tmp_path / "cl_state.json"
    optimizer.save_state(state_path)
    
    new_optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=ConstantLiar(base_acquisition=UpperConfidenceBound(kappa=2.576)),
        random_state=1,
        verbose=0
    )
    new_optimizer.load_state(state_path)
    
    verify_optimizers_match(optimizer, new_optimizer)

def test_integration_gp_hedge(target_func, pbounds, tmp_path):
    """Test save/load integration with GPHedge acquisition."""
    base_acquisitions = [
        UpperConfidenceBound(kappa=2.576),
        ProbabilityOfImprovement(xi=0.01),
        ExpectedImprovement(xi=0.01)
    ]
    acquisition_function = GPHedge(base_acquisitions=base_acquisitions)
    
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=2, n_iter=3)
    
    state_path = tmp_path / "gphedge_state.json"
    optimizer.save_state(state_path)
    
    new_base_acquisitions = [
        UpperConfidenceBound(kappa=2.576),
        ProbabilityOfImprovement(xi=0.01),
        ExpectedImprovement(xi=0.01)
    ]
    new_optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=GPHedge(base_acquisitions=new_base_acquisitions),
        random_state=1,
        verbose=0
    )
    new_optimizer.load_state(state_path)
    
    # Print new optimizer state
    print("\nNew Optimizer State:")
    print(f"GP random state: {new_optimizer._gp.random_state}")
    print(f"GP kernel params:\n{new_optimizer._gp.kernel_.get_params()}")
    print(f"Global random state: {np.random.get_state()[1][0]}")
    
    verify_optimizers_match(optimizer, new_optimizer)

def test_integration_constrained(target_func, pbounds, constraint, tmp_path):
    """Test save/load integration with constraints."""
    acquisition_function = ExpectedImprovement(xi=0.01)
    
    optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=acquisition_function,
        constraint=constraint,
        random_state=1,
        verbose=0
    )
    optimizer.maximize(init_points=2, n_iter=3)
    
    state_path = tmp_path / "constrained_state.json"
    optimizer.save_state(state_path)
    
    new_optimizer = BayesianOptimization(
        f=target_func,
        pbounds=pbounds,
        acquisition_function=ExpectedImprovement(xi=0.01),
        constraint=constraint,
        random_state=1,
        verbose=0
    )
    new_optimizer.load_state(state_path)
    
    verify_optimizers_match(optimizer, new_optimizer)
