#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Benchmark suite for joblib
#
# Author: Pierre Glaser
"""Benchmarking the impact of joblib's task batching strategy"""
import tempfile
import time

from joblib import Parallel, delayed
import numpy as np

from .common import Benchmark, sleep_noop


def bench_short_tasks(
    task_times,
    input_data_size=0,
    output_data_size=0,
    memmap_input=False,
    parallel_inst=None,
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

        parallel_inst(
            delayed(sleep_noop)(max(t, 0), input_data, output_data_size)
            for t in task_times
        )


class AutoBatchingSuite(Benchmark):
    repeat = 1
    number = 1
    warmup_time = 0

    # In practice, the input size does not influence the benchmarks very much:
    # The input is a numpy array. Small numpy arrays are pickled very fast and
    # do not incur significant overhead. Large numpy arrays can take time to be
    # pickled, but end up being memmapped, which only incurs a one-time
    # serialization cost, and thus will stop influence the batching after only
    # a few batches. Therefore, we fix this parameter for now.
    param_names = ["input_size", "n_jobs"]
    params = ([10000, 100000, 1000000], [1, 2, 4])
    parallel_parameters = dict(
        verbose=10, backend="loky", pre_dispatch="2*n_jobs"
    )
    bench_parameters = dict(
        output_data_size=int(1e5)  # output data size in bytes,
    )

    def setup(self, input_size, n_jobs):
        # Each benchmark is determined by its parameter set and the
        # duration profile of the tasks it has to execute. The following lines
        # defines a variety of task duration profiles.
        random_state = np.random.RandomState(42)
        high_variance = np.abs(
            random_state.normal(loc=0.000001, scale=0.01, size=5000)
        )

        low_variance = np.empty_like(high_variance)
        low_variance[:] = np.mean(high_variance)

        self.high_variance = high_variance
        self.low_variance = low_variance

        # Set up a cycling task duration pattern that the auto batching
        # feature should be able to roughly track. We use an even power of cos
        # to get only positive task durations with a majority close to zero
        # (only data transfer overhead).
        slow_time = 0.2
        positive_wave = np.cos(np.linspace(0, 6 * np.pi, 2000)) ** 8
        self.cyclic = positive_wave * slow_time

        # Simulate a situation where a few long tasks have to be executed, and
        # the first ones are cached. Because for the cached tasks, the apparent
        # compute time will seem very small, joblib will increase a lot the
        # batch size, which will potentially strangling workers.
        self.partially_cached = [1e-3] * 200 + [1] * 50

        self.parallel = Parallel(n_jobs=n_jobs, **self.parallel_parameters)

        # warm up the executor to mask worker-process creation overhead.
        self.parallel(delayed(time.sleep)(0.001) for _ in range(2 * n_jobs))

    def time_high_variance_no_trend(self, input_size, n_jobs):
        bench_short_tasks(
            self.high_variance,
            parallel_inst=self.parallel,
            input_data_size=input_size,
            **self.bench_parameters
        )

    time_high_variance_no_trend.pretty_name = (
        "Running time to complete tasks with high variance, untrended "
        "duration"
    )

    def time_low_variance_no_trend(self, input_size, n_jobs):
        bench_short_tasks(
            self.low_variance,
            parallel_inst=self.parallel,
            input_data_size=input_size,
        )
    time_low_variance_no_trend.pretty_name = (
        "Running time when computing tasks with low variance, untrended "
        "duration"
    )

    def time_cyclic_trend(self, input_size, n_jobs):
        return bench_short_tasks(
            self.cyclic,
            input_data_size=input_size,
            parallel_inst=self.parallel,
            **self.bench_parameters
        )

    time_cyclic_trend.pretty_name = (
        "Running time when computing tasks with cyclically trended duration"
    )

    def time_partially_cached(self, input_size, n_jobs):
        return bench_short_tasks(
            self.partially_cached,
            input_data_size=input_size,
            parallel_inst=self.parallel,
            **self.bench_parameters
        )

    time_partially_cached.pretty_name = (
        "Running time when computing long tasks, some of which already cached "
        "using joblib.Memory"
    )
