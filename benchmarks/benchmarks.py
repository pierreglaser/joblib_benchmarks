# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import os

from joblib import Parallel, delayed
import numpy as np

# Except for specific use cases, this will be the number of workers
# launched each time a Parallel(delayed(...)(...)) call is executed.
N_JOBS_MAX = os.cpu_count()

# Number of function calls submitted to each worker (on average)
# each time a Parallel(delayed(...)(...)) call is executed.
# Total number of function calls: N_JOBS_MAX*AVG_CALLS_PER_WORKERS
AVG_CALLS_PER_WORKERS = 2

N_FUNCTION_CALLS = AVG_CALLS_PER_WORKERS * N_JOBS_MAX


def make_arrays(shape, use_numpy):
    """wrapper around np.random.randn, with optional tolist()
    """
    arrays = np.random.randn(*shape)
    if not use_numpy:
        return arrays.tolist()
    else:
        return arrays


def compute_len(x):
    return len(x)


def make_bytes(size):
    return os.urandom(size)


def make_dict(size):
    return dict(zip(range(size), range(size)))


class TimeSuite:
    def time_array_as_input(self, shape, use_numpy):
        """make the parent create big arrays and send them to child processes

        For sufficiently large sizes (size>1e6 by default), memmapping will be
        automatically used
        """

        large_arrays = make_arrays((N_FUNCTION_CALLS, *shape),
                                   use_numpy)

        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(compute_len)(large_array) for large_array in large_arrays)

    time_array_as_input.param_names = ['shape', 'use_numpy']
    time_array_as_input.params = [
            ((10, 100), (100, 1000), (1000, 10000)),
            (True, False),
            ]

    def time_array_as_output(self, shape, use_numpy):
        """make child processes create big arrays and send it back

        For sufficiently large shapes, memmapping will be automatically used
        """

        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(make_arrays)(shape, use_numpy)
            for i in range(N_FUNCTION_CALLS))

    time_array_as_output.param_names = ['shape', 'use_numpy']
    time_array_as_output.params = [
            ((10, 100), (100, 1000), (1000, 10000)),
            (True, False),
            ]

    def time_dict_as_input(self, size):
        input_dict = make_dict(size)
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(compute_len)(input_dict) for i in range(N_FUNCTION_CALLS))

    time_dict_as_input.param_names = ['size']
    time_dict_as_input.params = [100, 1000, 10000]

    def time_bytes_as_input(self, size):
        input_bytes = make_bytes(size)
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(compute_len)(input_bytes) for i in range(N_FUNCTION_CALLS))

    time_bytes_as_input.param_names = ['size']
    time_bytes_as_input.params = [100, 1000, 10000]

    def time_bytes_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(make_bytes)(size) for i in range(N_FUNCTION_CALLS))

    time_bytes_as_output.param_names = ['size']
    time_bytes_as_output.params = [100, 1000, 10000]

    def time_dict_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(make_dict)(size) for i in range(N_FUNCTION_CALLS))

    time_dict_as_output.param_names = ['size']
    time_dict_as_output.params = [100, 1000, 10000]
