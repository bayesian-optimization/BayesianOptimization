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
    assert "..." in formatted

    # Test long float truncation
    long_float = 1234.5678901234
    formatted = logger._format_number(long_float)
    assert len(formatted) == logger._default_cell_size


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
    """Test the _step method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    # Register a point so we have something to log
    optimizer.register(params={"p1": 1.5, "p2": 2.5}, target=4.0)

    # Test default color
    step_str = logger._step(optimizer)
    assert "|" in step_str
    assert "1" in step_str  # iteration
    assert "4.0" in step_str  # target value

    # Test with custom color
    custom_color = Fore.RED
    step_str_colored = logger._step(optimizer, colour=custom_color)
    assert custom_color in step_str_colored


def test_header():
    """Test the _header method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    header_str = logger._header(optimizer)

    # Check if header contains expected column names
    assert "iter" in header_str
    assert "target" in header_str
    assert "p1" in header_str
    assert "p2" in header_str

    # Check if divider line is included
    assert "-" * 10 in header_str

    # Check with constrained logger
    constrained_logger = ScreenLogger(is_constrained=True)
    constrained_header = constrained_logger._header(optimizer)
    assert "allowed" in constrained_header


def test_is_new_max():
    """Test the _is_new_max method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    # No observations yet
    assert not logger._is_new_max(optimizer)

    # Add first observation
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    # The first time _is_new_max is called, it initializes the previous_max and returns False,
    # on the second call it should detect the first observation as a new max
    logger._is_new_max(optimizer)  # First call initializes _previous_max
    assert not logger._is_new_max(optimizer)  # Second call should still return False (not a new max)

    # Add lower observation (not a new max)
    optimizer.register(params={"p1": 0.5, "p2": 1}, target=1.5)
    assert not logger._is_new_max(optimizer)

    # Add higher observation (new max)
    optimizer.register(params={"p1": 2, "p2": 2}, target=4)
    assert logger._is_new_max(optimizer)  # Now we should have a new max


def test_update_tracker():
    """Test the _update_tracker method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)
    logger = ScreenLogger()

    # Initial state
    assert logger._iterations == 0
    assert logger._previous_max is None

    # Update with first observation
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    logger._update_tracker(optimizer)
    assert logger._iterations == 1
    assert logger._previous_max == 3
    assert logger._previous_max_params == {"p1": 1, "p2": 2}

    # Update with lower observation
    optimizer.register(params={"p1": 0.5, "p2": 1}, target=1.5)
    logger._update_tracker(optimizer)
    assert logger._iterations == 2
    assert logger._previous_max == 3  # Unchanged
    assert logger._previous_max_params == {"p1": 1, "p2": 2}  # Unchanged

    # Update with higher observation
    optimizer.register(params={"p1": 2, "p2": 2}, target=4)
    logger._update_tracker(optimizer)
    assert logger._iterations == 3
    assert logger._previous_max == 4  # Updated
    assert logger._previous_max_params == {"p1": 2, "p2": 2}  # Updated


def test_time_metrics():
    """Test the _time_metrics method."""
    logger = ScreenLogger()

    # First call initializes times
    time_str, total_elapsed, delta = logger._time_metrics()
    assert isinstance(time_str, str)
    assert isinstance(total_elapsed, float)
    assert isinstance(delta, float)
    assert delta <= 0.1  # First call should have very small delta

    # Subsequent call should show time difference
    import time

    time.sleep(0.01)  # Small delay to ensure time difference
    time_str2, total_elapsed2, delta2 = logger._time_metrics()
    assert total_elapsed2 > total_elapsed
    assert delta2 > 0


@patch("sys.stdout", new_callable=io.StringIO)
def test_log_optimization_start(mock_stdout):
    """Test the log_optimization_start method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    # Test with verbose=0 (should not print)
    logger = ScreenLogger(verbose=0)
    logger.log_optimization_start(optimizer)
    assert mock_stdout.getvalue() == ""

    # Test with verbose=1 (should print)
    logger.verbose = 1
    logger.log_optimization_start(optimizer)
    output = mock_stdout.getvalue()
    assert "iter" in output
    assert "target" in output
    assert "p1" in output
    assert "p2" in output


@patch("sys.stdout", new_callable=io.StringIO)
def test_log_optimization_step(mock_stdout):
    """Test the log_optimization_step method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    # Create logger with verbose=1 specifically, as this is the only verbose level
    # that doesn't print for non-max points according to the implementation:
    # if self._verbose != 1 or is_new_max:
    logger = ScreenLogger(verbose=1)

    # Clear any output that might have happened
    mock_stdout.truncate(0)
    mock_stdout.seek(0)

    # Register a point and log it
    optimizer.register(params={"p1": 1, "p2": 2}, target=3)
    logger._is_new_max(optimizer)  # Initialize previous_max

    # Register another point (not a new max)
    optimizer.register(params={"p1": 0.5, "p2": 1}, target=1.5)

    # Clear output buffer
    mock_stdout.truncate(0)
    mock_stdout.seek(0)

    # Log step - with verbose=1 and not a new max, it shouldn't print
    logger.log_optimization_step(optimizer)
    assert mock_stdout.getvalue() == ""

    # Test with verbose=2 (should print regular steps)
    logger.verbose = 2
    mock_stdout.truncate(0)
    mock_stdout.seek(0)

    # Log the same step again
    logger.log_optimization_step(optimizer)
    output = mock_stdout.getvalue()
    assert "1.5" in output  # Target value should be in output

    # Test with new max (should print even with verbose=1)
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.verbose = 1
    optimizer.register(params={"p1": 2, "p2": 2}, target=4)  # New max
    logger.log_optimization_step(optimizer)
    output = mock_stdout.getvalue()
    assert "4" in output  # Target value
    assert Fore.MAGENTA in output  # Should use magenta color for new max


@patch("sys.stdout", new_callable=io.StringIO)
def test_log_optimization_end(mock_stdout):
    """Test the log_optimization_end method."""
    optimizer = BayesianOptimization(target_func, PBOUNDS, random_state=1)

    # Initialize header length
    logger = ScreenLogger(verbose=2)
    logger.log_optimization_start(optimizer)

    # Test with verbose=0 (should not print)
    logger.verbose = 0
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.log_optimization_end(optimizer)
    assert mock_stdout.getvalue() == ""

    # Test with verbose=2 (should print)
    logger.verbose = 2
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    logger.log_optimization_end(optimizer)
    output = mock_stdout.getvalue()
    assert "=" in output  # Should contain the closing line
