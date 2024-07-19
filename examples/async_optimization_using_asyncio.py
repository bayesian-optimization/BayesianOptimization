# ruff: noqa: D401,D103
"""obtained from https://github.com/bayesian-optimization/BayesianOptimization/blob/v1.4.3/examples/async_optimization.py."""

from __future__ import annotations

import json
import asyncio
import threading
import secrets
import time
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any, Awaitable
from http.server import BaseHTTPRequestHandler, HTTPServer
from http.client import HTTPResponse
import multiprocessing
from urllib.request import Request, urlopen

from colorama import Fore

from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

OPTIMIZERS_CONFIG = (
    {"name": "optimizer 1", "colour": Fore.RED},
    {"name": "optimizer 2", "colour": Fore.GREEN},
    {"name": "optimizer 3", "colour": Fore.BLUE},
)
HOST = "localhost"
PORT = 10001
MAX_TRY_COUNT = 5


def black_box_function(x: float, y: float) -> float:
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, however, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its outputs values, as unknown.
    """
    time.sleep(secrets.randbelow(5) + 1)
    return -(x**2) - (y - 1) ** 2 + 1


class BayesianOptimizationHandler(BaseHTTPRequestHandler):
    _optimizer = BayesianOptimization(
        f=black_box_function,
        acquisition_function=UpperConfidenceBound(kappa=3),
        pbounds={"x": (-4, 4), "y": (-3, 3)},
    )
    _lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        length = self.headers.get("Content-Length", 0)
        post_data = self.rfile.read(int(length))

        params = json.loads(post_data)
        suggestion = self._register(params)
        result = json.dumps(suggestion).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(result)

    def _register(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            with suppress(KeyError):
                self._optimizer.register(
                    params=params["params"], target=params["target"]
                )
            return self._optimizer.suggest()


def run_server() -> None:
    server_address = (HOST, PORT)
    httpd = HTTPServer(
        server_address, BayesianOptimizationHandler, bind_and_activate=True
    )
    httpd.serve_forever()


def ping_to_server(max_try_count: int) -> bool:
    request = Request(f"http://{HOST}:{PORT}/", method="GET")
    response: HTTPResponse
    for _ in range(max_try_count):
        try:
            with urlopen(request, timeout=10) as response:
                return response.status == 200
        except URLError:
            time.sleep(1)
    return False


def reqeust_to_server(data: bytes) -> bytes:
    request = Request(
        f"http://{HOST}:{PORT}/",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    response: HTTPResponse
    with urlopen(request, data=data, timeout=10) as response:
        return response.read()


async def calculate_register_data(register_data: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(register_data).encode()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(reqeust_to_server, data)
        awaitable = asyncio.wrap_future(future)
        result = await awaitable
    return json.loads(result)


async def run_optimizer_per_config_process(register_data: dict[str, Any]) -> float:
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(black_box_function, **register_data["params"])
        awaitable = asyncio.wrap_future(future)
        return await awaitable


async def run_optimizer_per_config(
    config: dict[str, str],
    result: asyncio.Queue[tuple[str, float | None]],
) -> None:
    name, colour = config["name"], config["colour"]

    register_data: dict[str, Any] = {}
    max_target = None
    for _ in range(10):
        status = name + f" wants to register: {register_data}.\n"

        register_data["params"] = await calculate_register_data(register_data)
        target = register_data["target"] = await run_optimizer_per_config_process(
            register_data
        )

        if max_target is None or target > max_target:
            max_target = target

        status += name + f" got {target} as target.\n"
        status += name + f" will to register next: {register_data}.\n"
        print(colour + status, end="\n")

    await result.put((name, max_target))
    print(colour + name + " is done!", end="\n\n")


async def consume_results(
    result_queue: asyncio.Queue[tuple[str, float | None]], event: asyncio.Event
) -> None:
    while not event.is_set() or not result_queue.empty():
        result = result_queue.get()
        task = asyncio.create_task(result)
        try:
            result = await asyncio.wait_for(task, timeout=5)
        except (TimeoutError, asyncio.TimeoutError):
            task.cancel()
        else:
            print(result[0], f"found a maximum value of: {result[1]}")


async def wait(awaitable: Awaitable[Any], event: asyncio.Event) -> None:
    await awaitable
    event.set()


async def main() -> None:
    result_queue = asyncio.Queue(len(OPTIMIZERS_CONFIG))

    coro1, coro2, coro3 = (
        run_optimizer_per_config(config, result_queue) for config in OPTIMIZERS_CONFIG
    )
    gather_coro = asyncio.gather(coro1, coro2, coro3)
    event = asyncio.Event()

    gather_with_event = wait(gather_coro, event)
    consume = consume_results(result_queue, event)

    await asyncio.gather(gather_with_event, consume)


if __name__ == "__main__":
    server = multiprocessing.Process(target=run_server)
    server.start()
    try:
        if not ping_to_server(MAX_TRY_COUNT):
            raise RuntimeError("Server is not running.")
        asyncio.run(main())
    finally:
        server.terminate()
        server.join()
