from __future__ import annotations

import io
from unittest.mock import patch

from colorama import Fore

from bayes_opt import BayesianOptimization
from bayes_opt.logger import ScreenLogger


def target_func(**kwargs):
    """Simple target function for testing."""
    return sum(kwargs.values())


PBOUNDS = {"p1": (0, 10), "p2": (0, 10)}


def test_initialization():
    """Test logger initialization with default and custom parameters."""
    # Default parameters
    logger = ScreenLogger()
    assert logger.verbose == 2
    assert not logger.is_constrained

    # Custom parameters
    logger = ScreenLogger(verbose=0, is_constrained=True)
    assert logger.verbose == 0
    assert logger.is_constrained


def test_verbose_property():
    """Test the verbose property getter and setter."""
    logger = ScreenLogger(verbose=1)
    assert logger.verbose == 1

    logger.verbose = 0
    assert logger.verbose == 0

    logger.verbose = 2
    assert logger.verbose == 2


def test_is_constrained_property():
    """Test the is_constrained property getter."""
    logger = ScreenLogger(is_constrained=False)
    assert not logger.is_constrained

    logger = ScreenLogger(is_constrained=True)
    assert logger.is_constrained


def test_format_number():
    """Test the _format_number method."""
    logger = ScreenLogger()

    # Test integer formatting
    assert len(logger._format_number(42)) == logger._default_cell_size

    # Test float formatting with precision
    float_str = logger._format_number(3.14159)
    assert len(float_str) == logger._default_cell_size
    assert "3.14" in float_str  # default precision is 4

    # Test long integer truncation
    long_int = 12345678901234
    formatted = logger._format_number(long_int)
    assert len(formatted) == logger._default_cell_size
    assert formatted == "1.234e+13"

    # Test long float truncation
    long_float = 1234.5678901234
    formatted = logger._format_number(long_float)
    assert len(formatted) == logger._default_cell_size
    assert formatted == "1234.5678"

    # Test negative long float truncation
    long_float = -1234.5678901234
    formatted = logger._format_number(long_float)
    assert len(formatted) == logger._default_cell_size
    assert formatted == "-1234.567"

    # Test scientific notation truncation
    sci_float = 12345678901234.5678901234
    formatted = logger._format_number(sci_float)
    assert len(formatted) == logger._default_cell_size
    assert formatted == "1.234e+13"

    # Test negative scientific notation truncation
    sci_float = -12345678901234.5678901234
    formatted = logger._format_number(sci_float)
    assert len(formatted) == logger._default_cell_size
    assert formatted == "-1.23e+13"

    # Test long scientific notation truncation
    sci_float = -12345678901234.534e132
    formatted = logger._format_number(sci_float)
    assert len(formatted) == logger._default_cell_size
    assert formatted == "-1.2e+145"


def test_format_bool():
    """Test the _format_bool method."""
    logger = ScreenLogger()

    # Test True formatting
    true_str = logger._format_bool(True)
    assert len(true_str) == logger._default_cell_size
    assert "True" in true_str

    # Test False formatting
    false_str = logger._format_bool(False)
    assert len(false_str) == logger._default_cell_size
    assert "False" in false_str

    # Test with small cell size
    small_cell_logger = ScreenLogger()
    small_cell_logger._default_cell_size = 3
    assert small_cell_logger._format_bool(True) == "T  "
    assert small_cell_logger._format_bool(False) == "F  "


def test_format_str():
    """Test the _format_str method."""
    logger = ScreenLogger()

    # Test normal string
    normal_str = logger._format_str("test")
    assert len(normal_str) == logger._default_cell_size
    assert "test" in normal_str

    # Test long string truncation
    long_str = "this_is_a_very_long_string_that_should_be_truncated"
    formatted = logger._format_str(long_str)
    assert len(formatted) == logger._default_cell_size
    assert "..." in formatted


def test_step():
    """Test the _print_step method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    logger = ScreenLogger()

    # Register a point so we have something to log
    optimizer.register(params={"p1": 1.5, "p2": 2.5}, target=4.0)

    # Test default color
    step_str = logger._print_step(
        optimizer._space.keys, optimizer._space.res()[-1], optimizer._space.params_config
    )
    assert "|" in step_str
    assert "1" in step_str  # iteration
    assert "4.0" in step_str  # target value

    # Test with custom color
    custom_color = Fore.RED
    step_str_colored = logger._print_step(
        optimizer._space.keys, optimizer._space.res()[-1], optimizer._space.params_config, colour=custom_color
    )
    assert custom_color in step_str_colored


def test_print_header():
    """Test the _print_header method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    header_str = logger._print_header(optimizer._space.keys)

    # Check if header contains expected column names
    assert "iter" in header_str
    assert "target" in header_str
    assert "p1" in header_str
    assert "p2" in header_str

    # Check if divider line is included
    assert "-" * 10 in header_str

    # Check with constrained logger
    constrained_logger = ScreenLogger(is_constrained=True)
    constrained_header = constrained_logger._print_header(optimizer._space.keys)
    assert "allowed" in constrained_header


def test_is_new_max():
    """Test the _is_new_max method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    # No observations yet
    assert not logger._is_new_max(None)

    # Add first observation
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)

    current_max = optimizer.max
    logger._is_new_max(current_max)
    assert not logger._is_new_max(current_max)

    # Add lower observation (not a new max)
    optimizer.register(params={"p1": 0.5, "p2": 1}, target=1.5)
    assert not logger._is_new_max(optimizer.max)

    # Add higher observation (new max)
    optimizer.register(params={"p1": 2, "p2": 2}, target=4)
    assert logger._is_new_max(optimizer.max)


def test_update_tracker():
    """Test the _update_tracker method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    # Initial state
    assert logger._iterations == 0
    assert logger._previous_max is None

    # Update with first observation
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    logger._update_tracker(optimizer.max)
    assert logger._iterations == 1
    assert logger._previous_max == 3
    assert logger._previous_max_params == {"p1": 1, "p2": 2}

    # Update with lower observation
    optimizer.register(params={"p1": 0.5, "p2": 1}, target=1.5)
    logger._update_tracker(optimizer.max)
    assert logger._iterations == 2
    assert logger._previous_max == 3  # Unchanged
    assert logger._previous_max_params == {"p1": 1, "p2": 2}  # Unchanged

    # Update with higher observation
    optimizer.register(params={"p1": 2, "p2": 2}, target=4)
    logger._update_tracker(optimizer.max)
    assert logger._iterations == 3
    assert logger._previous_max == 4  # Updated
    assert logger._previous_max_params == {"p1": 2, "p2": 2}  # Updated


@patch("sys.stdout", new_callable=io.StringIO)
def test_log_optimization_start(mock_stdout):
    """Test the log_optimization_start method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    # Test with verbose=0 (should not print)
    logger = ScreenLogger(verbose=0)
    logger.log_optimization_start(optimizer._space.keys)
    assert mock_stdout.getvalue() == ""

    # Test with verbose=1 (should print)
    logger.verbose = 1
    logger.log_optimization_start(optimizer._space.keys)
    output = mock_stdout.getvalue()
    assert "iter" in output
    assert "target" in output
    assert "p1" in output
    assert "p2" in output


@patch("sys.stdout", new_callable=io.StringIO)
def test_log_optimization_step(mock_stdout):
    """Test the log_optimization_step method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    logger = ScreenLogger()

    # Create logger with verbose=1 specifically, as this is the only verbose level
    # that doesn't print for non-max points according to the implementation:
    # if self._verbose == 2 or is_new_max:
    logger.verbose = 1

    # Clear any output that might have happened
    mock_stdout.truncate(0)
    mock_stdout.seek(0)

    # Register a point and log it
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    current_max = optimizer.max
    logger._is_new_max(current_max)  # Initialize previous_max

    # For a point that is not a new max with verbose=1, should not print
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.log_optimization_step(
        optimizer._space.keys, optimizer._space.res()[-1], optimizer._space.params_config, optimizer.max
    )
    assert mock_stdout.getvalue() == ""  # Nothing printed for non-max point with verbose=1

    # Register a higher value, which should trigger output with verbose=1
    optimizer.register(params={"p1": 2, "p2": 2}, target=4)
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.log_optimization_step(
        optimizer._space.keys, optimizer._space.res()[-1], optimizer._space.params_config, optimizer.max
    )
    max_output = mock_stdout.getvalue()
    assert max_output != ""  # Something printed for new max point with verbose=1
    assert "4.0" in max_output  # Should show target value

    # Test with verbose=2 (should print even for non-max points)
    logger.verbose = 2
    optimizer.register(params={"p1": 1, "p2": 1}, target=1)
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.log_optimization_step(
        optimizer._space.keys, optimizer._space.res()[-1], optimizer._space.params_config, optimizer.max
    )
    non_max_output = mock_stdout.getvalue()
    assert non_max_output != ""  # Something printed for non-max point with verbose=2
    assert "1.0" in non_max_output  # Should show target value


@patch("sys.stdout", new_callable=io.StringIO)
def test_log_optimization_end(mock_stdout):
    """Test the log_optimization_end method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    # Initialize header length
    logger = ScreenLogger(verbose=2)
    logger.log_optimization_start(optimizer._space.keys)

    # Clear previous output
    mock_stdout.truncate(0)
    mock_stdout.seek(0)

    # Test with verbose=0 (should not print)
    logger.verbose = 0
    logger.log_optimization_end()
    assert mock_stdout.getvalue() == ""

    # Test with verbose=1 (should print)
    logger.verbose = 1
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.log_optimization_end()
    output = mock_stdout.getvalue()
    assert "=" in output  # Should show a line of equals signs
