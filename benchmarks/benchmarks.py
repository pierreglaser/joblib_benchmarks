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
def make_list(size):
    return list(range(size))


def make_dict(size):
    return dict(zip(range(size), range(size)))


class TimeSuite:

    # Numpy arrays benchmarks

    def time_np_array_as_input(self, size):
        """make the parent create big arrays and send them to child processes

        For sufficiently large sizes (size>1e6 by default), memmapping will be
        automatically used
        """

        arrays = np.random.randn(size)
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(np.sum)(array) for array in arrays)

    time_np_array_as_input.param_names = ['size']
    time_np_array_as_input.params = ([1000, 100000, 10000000], )

    def time_np_array_as_output(self, size):
        """make child processes create big arrays and send it back

        For sufficiently large shapes, memmapping will be automatically used
        """

        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(np.random.randn)(size) for i in range(N_FUNCTION_CALLS))

    time_np_array_as_output.param_names = ['size']
    time_np_array_as_output.params = ([1000, 100000, 10000000], )

    # def time_np_array_as_input_and_output(self, size):
    #     array = np.random.randn(size)

    #     # we reshape the array to the biggest possible square matrix, and
    #     # compute its eigenvalues in the child processes
    #     dim = np.floor(np.sqrt(len(e.size)))
    #     array = array.reshape(dim, dim)

    #     res = Parallel(n_jobs=N_JOBS_MAX)(
    #             delayed(np.linalg.eid)(array) for _ in
    #             range(N_FUNCTION_CALLS)
    #             )

    # time_np_array_as_input_and_output.param_names = ['size']
    # time_np_array_as_input_and_output.params = ([1000, 100000, 10000000],)

    # List benchmarks

    def time_list_as_input(self, size):
        """make the parent create big arrays and send them to child processes

        For sufficiently large sizes (size>1e6 by default), memmapping will be
        automatically used
        """

        input_list = make_list(size)
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(input_list) for _ in range(N_FUNCTION_CALLS))

    time_list_as_input.param_names = ['size']
    time_list_as_input.params = ([1000, 100000, 10000000], )

    def time_list_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(make_list)(size) for _ in range(N_FUNCTION_CALLS))

    time_list_as_output.param_names = ['size']
    time_list_as_output.params = ([1000, 100000, 10000000], )

    # Dict benchmarks

    def time_dict_as_input(self, size):
        input_dict = make_dict(size)
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(input_dict) for _ in range(N_FUNCTION_CALLS))

    time_dict_as_input.param_names = ['size']
    time_dict_as_input.params = ([1000, 100000, 10000000], )

    def time_dict_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(make_dict)(size) for _ in range(N_FUNCTION_CALLS))

    time_dict_as_output.param_names = ['size']
    time_dict_as_output.params = ([1000, 100000, 10000000], )

    # Bytes benchmarks

    def time_bytes_as_input(self, size):
        input_bytes = os.urandom(size)
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(input_bytes) for _ in range(N_FUNCTION_CALLS))

    time_bytes_as_input.param_names = ['size']
    time_bytes_as_input.params = ([1000, 100000, 10000000], )

    def time_bytes_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(os.urandom)(size) for _ in range(N_FUNCTION_CALLS))

    time_bytes_as_output.param_names = ['size']
    time_bytes_as_output.params = ([1000, 100000, 10000000], )
