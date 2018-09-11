# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
# from joblib import Parallel, delayed
import numpy as np
import os

from joblib import Parallel, delayed

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


def add_one(x):
    return x + 1


class TimeSuite:
    def time_parallel_dummy_call(self):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(add_one)(i) for i in range(1000)
        )
