import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from nlu_engine.utils.logger import create_logger

logger = create_logger(level="DEBUG")

execution_times: dict[str, float] = defaultdict(float)
total_calls_data = {"count": 0}

P = ParamSpec("P")
R = TypeVar("R")


def capture_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to measure and log the execution time of a function.

    This decorator tracks the total execution time for each decorated function
    and maintains a count of all decorated function calls.

    Args:
        func: The function whose execution time needs to be tracked.

    Returns:
        The wrapped function with execution time tracking.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        func_name = func.__name__
        execution_times[func_name] += elapsed
        total_calls_data["count"] += 1

        logger.info("-" * 20)
        logger.info(f"Time taken to run {func_name}: {elapsed:.4f} seconds")
        logger.info(f"Total decorated function calls so far: {total_calls_data['count']}")
        logger.info("=== CURRENT EXECUTION TIMES ===")
        logger.info(f"{get_execution_dict()}")
        logger.info("=" * 30)

        return result

    return wrapper


def get_execution_dict() -> dict[str, float]:
    """
    Retrieve the accumulated execution times for all tracked functions.

    Returns:
        A dictionary mapping function names to their total execution time.
    """
    return dict(execution_times)
