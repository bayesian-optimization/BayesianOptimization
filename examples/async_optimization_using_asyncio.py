# ruff: noqa: D401,D103
"""obtained from https://github.com/bayesian-optimization/BayesianOptimization/blob/v1.4.3/examples/async_optimization.py."""

from __future__ import annotations

import asyncio
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

from colorama import Fore

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

OPTIMIZERS_CONFIG = (
    {"name": "optimizer 1", "colour": Fore.RED},
    {"name": "optimizer 2", "colour": Fore.GREEN},
    {"name": "optimizer 3", "colour": Fore.BLUE},
)


def black_box_function(x: float, y: float) -> float:
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, however, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its outputs values, as unknown.
    """
    time.sleep(secrets.randbelow(5) + 1)
    return -(x**2) - (y - 1) ** 2 + 1


_ACQUISITION = UpperConfidenceBound(kappa=3)
_OPTIMIZER = BayesianOptimization(
    f=black_box_function, acquisition_function=_ACQUISITION, pbounds={"x": (-4, 4), "y": (-3, 3)}
)


async def calculate_register_data(register_data: dict[str, Any]) -> dict[str, Any]:
    with ThreadPoolExecutor(1) as pool:
        with suppress(KeyError):
            params, target = register_data["params"], register_data["target"]
            future = pool.submit(_OPTIMIZER.register, params=params, target=target)
            awaitable = asyncio.wrap_future(future)
            await awaitable
        future = pool.submit(_OPTIMIZER.suggest)
        awaitable = asyncio.wrap_future(future)
        return await awaitable


async def run_optimizer_per_config_process(register_data: dict[str, Any]) -> float:
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(black_box_function, **register_data["params"])
        awaitable = asyncio.wrap_future(future)
        return await awaitable


async def run_optimizer_per_config(
    config: dict[str, str], result: asyncio.Queue[tuple[str, float | None]]
) -> None:
    name, colour = config["name"], config["colour"]

    register_data: dict[str, Any] = {}
    max_target = None
    for _ in range(10):
        status = name + f" wants to register: {register_data}.\n"

        register_data["params"] = await calculate_register_data(register_data)
        target = register_data["target"] = await run_optimizer_per_config_process(register_data)

        if max_target is None or target > max_target:
            max_target = target

        status += name + f" got {target} as target.\n"
        status += name + f" will to register next: {register_data}.\n"
        print(colour + status, end="\n")

    await result.put((name, max_target))
    print(colour + name + " is done!", end="\n\n")


async def consume_results(result_queue: asyncio.Queue[tuple[str, float | None]]) -> None:
    while result_queue.empty():
        result = await result_queue.get()
        print(result[0], f"found a maximum value of: {result[1]}")


async def main() -> None:
    result_queue = asyncio.Queue(len(OPTIMIZERS_CONFIG))
    coro1, coro2, coro3 = (run_optimizer_per_config(config, result_queue) for config in OPTIMIZERS_CONFIG)
    gather = asyncio.gather(coro1, coro2, coro3)
    await gather
    await consume_results(result_queue)


if __name__ == "__main__":
    asyncio.run(main())
