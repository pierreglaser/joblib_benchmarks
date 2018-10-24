#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Benchmark suite for joblib
#
# Author: Pierre Glaser
"""Peak memory benchmarks for Parallel calls"""
import os

from joblib import Parallel, delayed
import numpy as np

from .common import N_FUNCTION_CALLS, N_JOBS_MAX
from .common import make_dict
from .common import Benchmark


class NpArraySuite(Benchmark):
    param_names = ['size', 'use_memmap']
    params = ([(100, 100), (300, 300), (1000, 1000)], [True, False])

    def setup(self, size, use_memmap):
        self.array = np.random.randn(*size)
        if use_memmap:
            # we force memmmaping for every array by setting the object size
            # threhsold above which memmaping is activated to its minimum
            # value (1bytes)
            self.max_nbytes = 1
        else:
            # setting memmap to None deactivates the memmaping option
            self.max_nbytes = None

    def peakmem_np_array_as_input(self, size, use_memmap):
        res = Parallel(
            n_jobs=N_JOBS_MAX, max_nbytes=self.max_nbytes)(
                delayed(np.sum)(self.array) for _ in range(N_FUNCTION_CALLS))

    peakmem_np_array_as_input.pretty_name = ('Parallel calls with numpy arrays'
                                             ' as inputs: peak memory usage')

    def peakmem_np_array_as_output(self, size, use_memmap):
        res = Parallel(
            n_jobs=N_JOBS_MAX,
            max_nbytes=self.max_nbytes)(delayed(
                lambda x: np.random.randn(*x))(size)
                for i in range(N_FUNCTION_CALLS))

    peakmem_np_array_as_output.pretty_name = (
        'Parallel calls with numpy '
        'arrays as outputs: peak memory usage')

    def peakmem_np_array_as_input_and_output(self, size, use_memmap):
        res = Parallel(
            n_jobs=N_JOBS_MAX,
            max_nbytes=self.max_nbytes)(delayed(
                lambda x: np.linalg.svd(x))(self.array)
                for _ in range(N_FUNCTION_CALLS))

    peakmem_np_array_as_input_and_output.pretty_name = (
        'Parallel calls with numpy arrays as outputs and inputs: peak'
        ' memory usage')


class ListSuite(Benchmark):
    param_names = ['size']
    params = ([10000, 100000, 1000000], )

    def setup(self, size):
        self.list = list(range(size))

    def peakmem_list_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.list) for _ in range(N_FUNCTION_CALLS))

    peakmem_list_as_input.pretty_name = ('Parallel calls with lists as'
                                         ' inputs: peak memory usage')

    def peakmem_list_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
                delayed(lambda x: list(range(x)))(size)
                for _ in range(N_FUNCTION_CALLS))

    peakmem_list_as_output.pretty_name = ('Parallel calls with lists as'
                                          ' outputs: peak memory usage')


class DictSuite(Benchmark):
    param_names = ['size']
    params = ([10000, 100000, 1000000], )

    def setup(self, size):
        self.dict = make_dict(size)

    def peakmem_dict_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.dict) for _ in range(N_FUNCTION_CALLS))

    peakmem_dict_as_input.pretty_name = ('Parallel calls with dicts as'
                                         ' inputs: peak memory usage')

    def peakmem_dict_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
                delayed(lambda x: dict(zip(range(x), range(x))))(size)
                for _ in range(N_FUNCTION_CALLS))

    peakmem_dict_as_output.pretty_name = ('Parallel calls with dicts as'
                                          ' outputs: peak memory usage')


class BytesSuite(Benchmark):
    param_names = ['size']
    params = ([10000, 100000, 1000000], )

    def setup(self, size):
        self.bytes = os.urandom(size)

    def peakmem_bytes_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.bytes) for _ in range(N_FUNCTION_CALLS))

    peakmem_bytes_as_input.pretty_name = ('Parallel calls with bytes as'
                                          ' inputs: peak memory usage')

    def peakmem_bytes_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(os.urandom)(size) for _ in range(N_FUNCTION_CALLS))

    peakmem_bytes_as_output.pretty_name = ('Parallel calls with bytes as'
                                           ' outputs: peak memory usage')
