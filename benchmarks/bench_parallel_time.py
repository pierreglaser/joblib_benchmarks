#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Benchmark suite for joblib
#
# Author: Pierre Glaser
"""Running time benchmarks for Parallel calls"""
import os

from joblib import Parallel, delayed
import numpy as np

from .common import N_FUNCTION_CALLS, N_JOBS_MAX
from .common import compute_eigen, make_dict, make_list
from .common import Benchmark


class NpArraySuite:
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

    def time_np_array_as_input(self, size, use_memmap):
        """make the parent create big arrays and send them to child processes

        For sufficiently large sizes (size>1e6 by default), memmapping will be
        automatically used
        """

        res = Parallel(
            n_jobs=N_JOBS_MAX, max_nbytes=self.max_nbytes)(
                delayed(np.sum)(self.array) for _ in range(N_FUNCTION_CALLS))

    time_np_array_as_input.pretty_name = ('Parallel calls with numpy arrays as'
                                          ' inputs: running time')

    def time_np_array_as_output(self, size, use_memmap):
        res = Parallel(
            n_jobs=N_JOBS_MAX,
            max_nbytes=self.max_nbytes)(delayed(
                lambda x: np.random.randn(*x))(size)
                for i in range(N_FUNCTION_CALLS))

    time_np_array_as_output.pretty_name = ('Parallel calls with numpy arrays'
                                           ' as outputs: running time')

    def time_np_array_as_input_and_output(self, size, use_memmap):
        res = Parallel(
            n_jobs=N_JOBS_MAX,
            max_nbytes=self.max_nbytes)(delayed(
                lambda x: np.linalg.svd(x))(self.array)
                for _ in range(N_FUNCTION_CALLS))

    time_np_array_as_input_and_output.pretty_name = (
        'Parallel calls with numpy arrays as outputs and inputs: running'
        ' time')


class ListSuite:
    param_names = ['size']
    params = ([10000, 100000, 1000000], )
    timeout = 180
    processes = 1
    number = 2
    repeat = 1

    def setup(self, size):
        self.list = list(range(size))

    def time_list_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.list) for _ in range(N_FUNCTION_CALLS))

    time_list_as_input.pretty_name = ('Parallel calls with lists as'
                                      ' inputs: running time')

    def time_list_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
                delayed(lambda x: list(range(x)))(size)
                for _ in range(N_FUNCTION_CALLS))

    time_list_as_output.pretty_name = ('Parallel calls with lists as'
                                       ' outputs: running time')


class BytesSuite:
    param_names = ['size']
    params = ([10000, 100000, 1000000], )
    timeout = 180
    processes = 1
    number = 2
    repeat = 1

    def setup(self, size):
        self.bytes = os.urandom(size)

    def time_bytes_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.bytes) for _ in range(N_FUNCTION_CALLS))

    time_bytes_as_input.pretty_name = ('Parallel calls with bytes as'
                                       ' inputs: running time')

    def time_bytes_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(os.urandom)(size) for _ in range(N_FUNCTION_CALLS))

    time_bytes_as_output.pretty_name = ('Parallel calls with bytes as'
                                        ' outputs: running time')


class DictSuite:
    param_names = ['size']
    params = ([10000, 100000, 1000000], )

    def setup(self, size):
        self.dict = make_dict(size)

    def time_dict_as_input(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
            delayed(len)(self.dict) for _ in range(N_FUNCTION_CALLS))

    time_dict_as_input.pretty_name = ('Parallel calls with dicts as'
                                      ' inputs: running time')

    def time_dict_as_output(self, size):
        res = Parallel(n_jobs=N_JOBS_MAX)(
                delayed(lambda x: dict(zip(range(x), range(x))))(size)
                for _ in range(N_FUNCTION_CALLS))

    time_dict_as_output.pretty_name = ('Parallel calls with dicts as'
                                       ' outputs: running time')
