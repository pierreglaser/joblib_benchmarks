#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Benchmark suite for joblib
#
# Author: Pierre Glaser
"""Benchmarking the inpact of joblib's task batching strategy"""
import tempfile
import time

from joblib import Parallel, delayed
import numpy as np

from .common import Benchmark, sleep_noop


def bench_short_tasks(
    task_times,
    n_jobs=2,
    batch_size="auto",
    pre_dispatch="2*n_jobs",
    verbose=True,
    input_data_size=0,
    output_data_size=0,
    backend="loky",
    memmap_input=False,
    eta=None
):

    with tempfile.NamedTemporaryFile() as temp_file:
        if input_data_size:
            # Generate some input data with the required size
            if memmap_input:
                temp_file.close()
                input_data = np.memmap(
                    temp_file.name,
                    shape=input_data_size,
                    dtype=np.byte,
                    mode="w+",
                )
                input_data[:] = 1
            else:
                input_data = np.ones(input_data_size, dtype=np.byte)
        else:
            input_data = None

        with Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            batch_size=batch_size,
            backend=backend,
            max_nbytes=0,
            eta=eta
        ) as p:
            # Call the Parallel object from a context_manager to manage the
            # backend ourselves and skip cleanup rountines at the end of the
            # call that, among other things, reset the effective_batch_size of
            # the backend to 1.

            t0 = time.time()
            p(
                delayed(sleep_noop)(max(t, 0), input_data, output_data_size)
                for t in task_times
            )
            total_time = time.time() - t0

            batch_infos = p._backend._batch_info

        # compute compute time per batch. If the batch size selection algorithm
        # is good and the compute time of each task is not too chaotic, this
        # compute time should stabilize between MIN_BATCH_DURATION and
        # MAX_BATCH_DURATION.
        batch_durations = []
        first_task_idx = 0
        batch_infos = sorted(batch_infos, key=lambda x: x[0])
        for info in batch_infos:
            batch_size = info[1]
            last_task_idx = first_task_idx + info[1]
            batch_duration = sum(task_times[first_task_idx:last_task_idx])
            info.append(batch_duration)
            first_task_idx = last_task_idx

    return (batch_infos, total_time)


class AutoBatchingSuite(Benchmark):
    repeat = 1
    number = 1
    warmup_time = 0

    param_names = ["size", "eta"]
    params = ([10000, 100000, 1000000][:1], [1, 0.8, 0.5, 0.2][1:2])
    bench_parameters = dict(
        output_data_size=int(1e5),  # output data size in bytes
        n_jobs=4,
        verbose=10,
    )

    def setup(self, size, eta):
        random_state = np.random.RandomState(42)
        high_variance = np.abs(
            random_state.normal(loc=0.000001, scale=0.01, size=5000)
        )

        low_variance = np.empty_like(high_variance)
        low_variance[:] = np.mean(high_variance)

        self.high_variance = high_variance
        self.low_variance = low_variance

        # one has a cycling task duration pattern that the auto batching
        # feature should be able to roughly track. We use an even power of cos
        # to get only positive task durations with a majority close to zero
        # (only data transfer overhead). The shuffle variant should not
        # oscillate too much and still approximately have the same total run
        # time.
        slow_time = 0.2
        positive_wave = np.cos(np.linspace(0, 2 * np.pi, 1000)) ** 8
        self.cyclic = positive_wave * slow_time

        # Simulate a situation where a few long tasks have to be executed, and
        # the first ones are cached. Because for the cached task, the apparent
        # compute time will seem very small, joblib will increase a lot the
        # batch size, which will potentially make the workers starve.
        self.partially_cached = [1e-3] * 200 + [1] * 50

    def time_high_variance_no_trend(self, size, eta):
        bench_short_tasks(
            self.high_variance, **self.bench_parameters, input_data_size=size
        )

    time_high_variance_no_trend.pretty_name = (
        "running time to complete tasks with high variance, untrended running "
        "time"
    )

    def time_low_variance_no_trend(self, size, eta):

        bench_short_tasks(
            self.low_variance, **self.bench_parameters, input_data_size=size,
            eta=eta
        )

    def track_high_variance_no_trend(self, size, eta):
        return bench_short_tasks(
            self.high_variance, **self.bench_parameters, input_data_size=size,
            eta=eta
        )

    track_high_variance_no_trend.pretty_name = (
        "effective batch size when running tasks with high variance, untrended"
        " running time"
    )

    def track_low_variance_no_trend(self, size, eta):
        return bench_short_tasks(
            self.low_variance, **self.bench_parameters, input_data_size=size,
            eta=eta
        )

    track_low_variance_no_trend.pretty_name = (
        "effective batch size when running tasks with low variance, untrended"
        " running time"
    )

    def track_cyclic_trend(self, size, eta):
        return bench_short_tasks(
            self.cyclic, **self.bench_parameters, input_data_size=size, eta=eta
        )

    track_cyclic_trend.pretty_name = (
        "effective batch size when running tasks cyclically trended running"
        " time"
    )

    def track_partially_cached(self, size, eta):
        return bench_short_tasks(
            self.partially_cached,
            **self.bench_parameters,
            input_data_size=size,
            eta=eta
        )

    track_partially_cached.pretty_name = (
        "effective batch size when running long task, with the first ones"
        " simply retrieved from a cache"
    )
