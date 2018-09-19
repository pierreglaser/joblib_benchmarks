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


# Small helper functions as it is not possible to create basic instances of
# list/dict of a specific size using a single function call
def make_dict(size):
    return dict(zip(range(size), range(size)))


def compute_eigen(arr):
    """reshape a 1-dim array to a square matrix and compute its eigenvalues

    :param arr: a 1-dimensional array

    In order for arr to be reshaped to a square matrix, it must be a perfect
    # square. For this reason,
    - we find the closest exact square to its number of elements (dim**2)
    - the we subset arr it to its first dim**2 values
    - finally, we do the appropriate reshaping with a dimension of dim
    """

    dim = int(np.sqrt(arr.size))
    arr = arr[:dim**2]
    square_matrix = arr.reshape(dim, dim)
    return np.linalg.svd(square_matrix)


# The benchmark routine of asv can be summarized as follow:
# for b in benchmark:                        # loop 1
#     for i in range(b.repeat):              # loop 2
#         for args in params:                # loop 3
#             b.setup(*arg)
#             for function in b:             # loop 4
#                 for n in range(b.number):  # loop 5
#                     function(*arg)
#             b.teardown(*arg)
#
# number and repeat attributes differ in the sense that a setup and teardown
# call is run between two iterations of loop 2, in opposition with loop 5.
# For most cases here, we will use the number attribute


class NpArrayTimeSuite:
    param_names = ['size']
    params = ([1000, 100000, 10000000], )

    # if a benchmark does not return anything after 180 it will fail
    # automatically
    timeout = 180
    processes = 1
    number = 2
    repeat = 1

    def setup(self, size):
        self.array = np.random.randn(size)

    def time_np_array_as_input(self, size):
        """make the parent create big arrays and send them to child processes

        For sufficiently large sizes (size>1e6 by default), memmapping will be
        automatically used
        """

        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(np.sum)(self.array) for _ in range(N_FUNCTION_CALLS))

    def time_np_array_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(np.random.randn)(size) for i in range(N_FUNCTION_CALLS))

    def time_np_array_as_input_and_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(delayed(compute_eigen)(self.array)
                                          for _ in range(N_FUNCTION_CALLS))


# List benchmarks
class ListTimeSuite:
    param_names = ['size']
    params = ([1000, 100000, 10000000], )
    timeout = 180
    processes = 1
    number = 2
    repeat = 1

    def setup(self, size):
        self.list = list(range(size))

    def time_list_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.list) for _ in range(N_FUNCTION_CALLS))

    def time_list_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(lambda x: list(range(x)))(size)
            for _ in range(N_FUNCTION_CALLS))


# Dict benchmarks
class DictTimeSuite:
    param_names = ['size']
    params = ([1000, 100000, 10000000], )
    timeout = 180
    processes = 1
    number = 2
    repeat = 1

    def setup(self, size):
        self.dict = make_dict(size)

    def time_dict_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.dict) for _ in range(N_FUNCTION_CALLS))

    def time_dict_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(make_dict)(size) for _ in range(N_FUNCTION_CALLS))


# Bytes benchmarks
class BytesBenchmark:
    param_names = ['size']
    params = ([1000, 100000, 10000000], )
    timeout = 180
    processes = 1
    number = 2
    repeat = 1

    def setup(self, size):
        self.bytes = os.urandom(size)

    def time_bytes_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.bytes) for _ in range(N_FUNCTION_CALLS))

    def time_bytes_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(os.urandom)(size) for _ in range(N_FUNCTION_CALLS))
