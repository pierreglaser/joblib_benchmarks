import datetime
import time
import json
import os
from math import log10
import socket

import numpy as np
import pandas as pd
import matplotlib
import re
import matplotlib.pyplot as plt

from asv.benchmarks import Benchmarks
from asv.config import Config
from asv.commands.publish import Publish

HOME = os.environ.get('HOME')
hostname = socket.gethostname()


def create_benchmark_dataframe(concat_benchmarks=False):
    repo_dirname = os.path.dirname(__file__)
    config_path = os.path.join(repo_dirname, 'asv.conf.json')
    config = Config.load(config_path)

    benchmarks = Benchmarks.load(config)

    results = {}

    for single_env_result in Publish.iter_results(config, benchmarks):
        single_env_result_df = {}
        for b_name, params in single_env_result._benchmark_params.items():
            _benchmark = benchmarks[b_name]
            b_type, b_unit, param_names = (_benchmark['type'],
                                           _benchmark['unit'],
                                           _benchmark['param_names'])
            filename, classname, benchname = b_name.split('.')

            mi = pd.MultiIndex.from_product(params, names=param_names)

            # if len(mi) != len(single_env_result._results[b_name]):
            #     print('something went wrong with {} {}'.format(b_name,
            #           single_env_result._commit_hash))
            #     continue
            _results = pd.Series(single_env_result._results[b_name], index=mi)

            # benchmark level metadata
            benchmark_level_metadata = (b_type, benchname, classname, filename)
            single_env_result_df[benchmark_level_metadata] = _results

        if concat_benchmarks is True:
            single_env_result_df = pd.concat(
                single_env_result_df,
                names=['type', 'name', 'class', 'file'],
                axis=0)
        elif concat_benchmarks == 'class':
            unique_classes = np.unique(
                [b[2] for b in single_env_result_df.keys()])

            bench_by_class = {}
            for c in unique_classes:
                class_bench = {
                    metadata: bench_df
                    for metadata, bench_df in single_env_result_df.items()
                    if metadata[2] == c
                }
                bench_by_class[c] = pd.concat(
                    class_bench,
                    names=['type', 'name', 'class', 'file'],
                    axis=0)
            single_env_result_df = bench_by_class
        else:
            single_env_result_df = dict(
                zip(benchmark_level_metadatas, single_env_result_df))

        single_env_metadata = (single_env_result._params['python'],
                               single_env_result._commit_hash,
                               single_env_result._date)

        results[single_env_metadata] = single_env_result_df
        # metadata.append(single_env_metadata)

    if len(results) == 0:
        print('no benchmark found')
        return

    if concat_benchmarks is True:
        all_benchmark_results = pd.concat(
            results, names=['python', 'commit_hash', 'date'], axis=0)
        return all_benchmark_results
    elif concat_benchmarks == 'class':
        all_benchmark_results = {}
        for c in unique_classes:
            all_benchmark_results[c] = pd.concat(
                {
                    commit_metatada: r[c]
                    for commit_metatada, r in results.items()
                },
                names=['python', 'commit_hash', 'date'])
    else:
        all_benchmark_results = {}
        for benchmark_metadata in benchmark_level_metadatas:
            all_benchmark_results[benchmark_metadata] = pd.concat(
                {
                    commit_metatada: r[benchmark_metadata]
                    for commit_metatada, r in zip(metadata, results)
                },
                names=['python', 'commit_hash', 'date'])

    return all_benchmark_results


if __name__ == "__main__":
    all_bench = create_benchmark_dataframe(concat_benchmarks='class')
