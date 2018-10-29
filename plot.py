import json
import os
from math import log

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import benchmarks

from sum_up_results import create_benchmark_dataframe


def plot_results(df,
                 title=None,
                 x_label=None,
                 y_label=None,
                 hue_label=None,
                 row_label=None,
                 col_label=None):
    # first get the number of value per parameters
    df = df.to_frame('time').reset_index()
    # replace '' by classic pickler
    df = df.loc[df['n_jobs'] == '2', :]
    df = df.replace("''", 'pickle')
    df = df.replace("'cloudpickle'", 'cloudpickle')

    df = df[[x_label, y_label, row_label, col_label, hue_label]]
    n_values = df.nunique()

    # get only interesting values
    # specifiy aesthetics parameters

    colors = {'pickle': 'red', 'cloudpickle': 'orange'}

    use_errorbars = False

    # create a subplot grid with the correct shape
    f, axs = plt.subplots(
        nrows=n_values[row_label],
        ncols=n_values[col_label],
        squeeze=False,
        figsize=(12, 8),
        sharex=False,
        sharey='row')
    f.suptitle(title, x=0.5, y=0.92)

    # create the legend
    f.legend(
        handles=[Patch(color=v, label=k) for k, v in colors.items()],
        loc='upper center',
        ncol=2)

    df = df.sort_values(by=x_label)
    # loop over all groups
    by_rowsandcol = df.groupby([row_label, col_label])
    targets = zip(by_rowsandcol, axs.flatten())
    for group, ax in targets:
        n, g = group

        infos = zip((row_label, col_label), n)
        text = '\n'.join(['{}: {}'.format(*vals) for vals in infos])
        ax.text(0.1, 0.7, text, transform=ax.transAxes, bbox=dict(alpha=0.2))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_yscale('log')

        # re comput n_values for the specific group
        n_values = g.nunique()
        x_ticks = np.arange(n_values[x_label])
        ax.set_xticklabels(g[x_label].unique())
        ax.set_xticks(x_ticks)
        width = 0.2
        hue_no = 0
        for hue, subgroup in g.groupby(hue_label):
            x = subgroup[x_label].tolist()
            y = subgroup[y_label].tolist()

            by_xlabel = subgroup.groupby(x_label)
            if use_errorbars:
                ax.fill_between(
                    subgroup[x_label].unique(),
                    by_xlabel[y_label].min(),
                    by_xlabel[y_label].max(),
                    color=colors[hue],
                    alpha=0.3)
            else:
                bars = by_xlabel.mean()
                ax.bar(
                    x_ticks + hue_no * width,
                    bars[y_label],
                    width,
                    alpha=0.5,
                    color=colors[hue])
                # show the points for each bar
                xtick = 0
                for n, g in by_xlabel:
                    ax.scatter(
                        [xtick + hue_no * width] * len(g),
                        g[y_label],
                        # 'b+',
                        c='black',
                        s=20,
                        marker='_',
                        alpha=0.5)
                    xtick += 1
            hue_no += 1

    return f


if __name__ == "__main__":
    all_dfs = create_benchmark_dataframe(concat_benchmarks='class')
    # all_dfs = create_benchmark_dataframe(concat_benchmarks)
    labels = {
        'MakeRegressionDataBench':
        dict(
            y_label='time',
            x_label='n_samples',
            hue_label='pickler',
            col_label='backend',
            row_label='name'),
        'CaliforniaHousingBench':
        dict(
            # set of labels for text benchs
            y_label='time',
            x_label='backend',
            hue_label='pickler',
            col_label='file',
            row_label='name'),
        'TwentyDataBench':
        dict(
            # set of labels for text benchs
            y_label='time',
            x_label='backend',
            hue_label='pickler',
            col_label='file',
            row_label='name')
    }
    for benchmark_name, benchmark_df in all_dfs.items():
        # first get the number of value per parameters
        f = plot_results(
            benchmark_df, title=benchmark_name, **labels[benchmark_name])
        f.savefig('{}.png'.format(benchmark_name))
