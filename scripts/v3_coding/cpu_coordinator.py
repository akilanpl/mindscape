"""Independent-request batching with cooperative deadline/interrupt draining."""

import queue
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor

from mindscape.coding.batching import EpisodeCoder


def evaluate_batch(coder, jobs, solve, save, batch_size=4, deadline=None):
    requests = queue.Queue()
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=batch_size)
    futures = []
    delivered = set()
    completed = 0
    failure = None
    ready = []
    coder.deadline = deadline

    def episode(job):
        if stop.is_set():
            return None
        return solve(job, EpisodeCoder(requests, coder.revision))

    def deliver():
        nonlocal completed, failure
        for index, future in enumerate(futures):
            if index in delivered or not future.done():
                continue
            delivered.add(index)
            try:
                row = future.result()
            except (TimeoutError, CancelledError):
                continue
            except Exception as error:  # noqa: BLE001 - propagate after draining all consumers
                failure = failure or error
                stop.set()
                continue
            if row is not None:
                save(row)
                completed += 1

    try:
        futures = [pool.submit(episode, job) for job in jobs]
        while len(delivered) < len(futures):
            if deadline is not None and time.time() >= deadline:
                stop.set()
            ready = []
            try:
                ready.append(requests.get(timeout=0.02))
                until = time.monotonic() + 0.02
                while len(ready) < batch_size and time.monotonic() < until:
                    try:
                        ready.append(requests.get(timeout=max(0.001, until - time.monotonic())))
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            if ready:
                if stop.is_set():
                    for request in ready:
                        request.future.set_exception(TimeoutError("Study stopped"))
                else:
                    try:
                        values = coder.generate_batch([r.args for r in ready])
                        for request, value in zip(ready, values, strict=True):
                            request.future.set_result(value)
                    except Exception as error:  # noqa: BLE001 - release every waiting consumer
                        failure = error
                        stop.set()
                        for request in ready:
                            request.future.set_exception(error)
            deliver()
    except KeyboardInterrupt as error:
        failure = error
    finally:
        stop.set()
        for request in ready:
            if not request.future.done():
                request.future.set_exception(TimeoutError("Study stopped"))
        for future in futures:
            if not future.running():
                future.cancel()
        # Every active client waiting on a response must be released before joining.
        while not all(f.done() for f in futures):
            try:
                request = requests.get(timeout=0.02)
                if not request.future.done():
                    request.future.set_exception(TimeoutError("Study stopped"))
            except queue.Empty:
                pass
            deliver()
        deliver()
        pool.shutdown(wait=True, cancel_futures=True)
    if failure is not None and not isinstance(failure, TimeoutError):
        raise failure
    return completed
