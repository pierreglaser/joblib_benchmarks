# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os

from joblib import Parallel, delayed
import numpy as np

# except for specific use cases, this will be the
# number of worker launched each time a Parallel(delayed) call
# is executed
N_JOBS_MAX = os.cpu_count()

# number of function calls submitted to each worker (on average)
# each time a Parallel(delayed(...)) call is done
# total number of function calls: N_JOBS_MAX*AVG_CALLS_PER_WORKERS
AVG_CALLS_PER_WORKERS = 2

N_FUNCTION_CALLS = AVG_CALLS_PER_WORKERS * N_JOBS_MAX


def make_arrays(n, shape, use_numpy=True):
    arrays = np.random.randn(n, *shape)
    if not use_numpy:
        return arrays.tolist()
    else:
        return arrays


def return_one(x):
    return 1


def add_one(x):
    return x + 1


class TimeSuite:
    def time_parallel_dummy_call(self):
        res = Parallel(
            n_jobs=N_JOBS_MAX, backend='loky')(
                delayed(add_one)(i) for i in range(N_FUNCTION_CALLS))

    def time_large_array_as_input_and_small_output(self, use_numpy, shape):
        """benchark the time of shipping a numpy array to a child process

        for sufficiently big arrays (size>1e6 for joblib by default)
        """

        large_arrays = make_arrays(
            N_FUNCTION_CALLS, shape, use_numpy=use_numpy)

        res = Parallel(
            n_jobs=N_JOBS_MAX, backend='loky')(delayed(return_one)(large_array)
                                               for large_array in large_arrays)

    time_large_array_as_input_and_small_output.param_names = [
        'use_numpy', 'shape'
    ]
    time_large_array_as_input_and_small_output.params = [
            (True, False),
            ((10, 100), (100, 1000), (1000, 10000))
            ]  # yapf: disable

    def time_small_input_and_large_array_as_output(use_numpy, shape):
        """benchark the time of shipping a numpy array back to
        the parent process

        for sufficiently big arrays (size>1e6 for joblib by default)
        """

        large_arrays = make_arrays(
            N_FUNCTION_CALLS, shape, use_numpy=use_numpy)

        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(return_one)(large_array) for large_array in large_arrays)

    time_small_input_and_large_array_as_output.param_names = [
        'use_numpy', 'shape'
    ]
    time_small_input_and_large_array_as_output.params = [
            (True, False),
            ((10, 100), (100, 1000), (1000, 10000))
            ]  # yapf: disable
