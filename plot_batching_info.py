import pandas as pd
import matplotlib.pyplot as plt

from asv_to_pandas import create_benchmark_dataframe

BENCHMARK_INFO_COLS = [
    "batch_idx",
    "batch_size",
    "total_duration",
    "smoothed_duration",
    "used_to_compute_bs",
    "last_influential_batch_idx",
    "compute_duration",
]

hash_to_branch_map = {
    "c3610a79e399c8c207cce90cfc088cfcf714e155": "master",
    "be2e1da190c2638d6a80fe00751c4e8011dbd8e8": "master-slow-increase",
    "9362fb42c9150066072685fb54d9df7afb0af936": "synchronize-batch-operations",
    "b84fa5a601295b1f86bbf9501b3232c92b9806e7": "synchronized-slow-increase"

}

benchmark_names = [
    "track_cyclic_trend",
    "track_high_variance_no_trend",
    "track_low_variance_no_trend",
    "track_partially_cached",
]


def aggregate_benchmark_dataframes():
    df = create_benchmark_dataframe(group_by="file")["bench_auto_batching"]

    # each line of df contains an array of batch size records
    batch_info_dfs = {}
    times = []

    for bench_metadata, batch_info in df.iteritems():
        batch_info, total_time = batch_info
        batch_info_df = pd.DataFrame(batch_info, columns=BENCHMARK_INFO_COLS)
        batch_info_dfs[bench_metadata] = batch_info_df

        times.append(total_time)

    batch_info_dfs = pd.concat(batch_info_dfs, names=df.index.names, axis=0)
    batch_info_dfs.used_to_compute_bs *= 1
    time_df = pd.Series(times, df.index).to_frame("time")

    # quick hack to use branch name instead of commit hash
    batch_info_dfs = (
        batch_info_dfs.reset_index("commit_hash")
        .replace(hash_to_branch_map)
        .set_index("commit_hash", append=True)
    )
    batch_info_dfs = batch_info_dfs.reset_index(
        ["version", "date", "class", "type"], drop=True
    )

    time_df = (
        time_df.reset_index("commit_hash")
        .replace(hash_to_branch_map)
        .set_index("commit_hash", append=True)
    )
    time_df = time_df.reset_index(
        ["version", "date", "class", "type"], drop=True
    )

    return batch_info_dfs, time_df


def reformat_single_benchmark_dataframe(df):
    df = df.set_index("batch_idx")
    assert not df.index.duplicated().sum(), "index contains duplicates!"
    df["task_idx"] = df.batch_size.cumsum()
    df.set_index("task_idx", inplace=True)

    # make batch size graphs more intuitive
    df = df.shift(-1).ffill()

    df_mean = df.copy()
    df_mean.total_duration /= df_mean.batch_size
    df_mean.smoothed_duration /= df_mean.batch_size
    df_mean.compute_duration /= df_mean.batch_size

    df_cumsum = df.cumsum()

    return df_mean, df_cumsum


def plot_benchmark_dfs(df_mean, df_cumsum, title, extra_text):
    f, axs = plt.subplots(nrows=2)
    for ax, df in zip(axs, [df_mean, df_cumsum]):
        # forget about the first noisy observations
        df = df.iloc[15:,]
        ax1 = df[
            ["total_duration", "compute_duration", "smoothed_duration"]
        ].plot(ax=ax, style="o-", ms=3)
        ax2 = df["batch_size"].plot(ax=ax, secondary_y=True)
        ax3 = df.used_to_compute_bs.plot(ax=ax, secondary_y=True)

        ax.legend(
            ax1.lines + ax2.lines + ax3.lines,
            [
                "total_duration",
                "compute_duration",
                "smoothed_duration",
                "batch_size",
                "used to compute bs",
            ],
        )
        ax.set_title(title)
        ax.text(0.5, 0.9, extra_text, transform=ax.transAxes)


def compare(bench_name, branch_one, branch_two):
    batch_df, time_df = aggregate_benchmark_dataframes()
    batch_df_dev = batch_df.xs(
        [branch_one, bench_name],
        level=["commit_hash", "name"],
        drop_level=False,
    )

    batch_df_master = batch_df.xs(
        [branch_two, bench_name], level=["commit_hash", "name"], drop_level=False
    )
    return batch_df_dev, batch_df_master


if __name__ == "__main__":
    batch_df, time_df = aggregate_benchmark_dataframes()
    batch_df_dev, batch_df_master = compare(
        # "track_cyclic_trend"
        "track_partially_cached",
        "synchronized-slow-increase",
        "master-slow-increase",
    )

    # by_batch_branch_and_benchmark = batch_df.groupby(
    #     ["commit_hash", "name"], axis=0
    # )

    # for name, single_bench_df in by_batch_branch_and_benchmark:
    #     branch_name, bench_name = name
    #     df_mean, df_cumsum = reformat_single_benchmark_dataframe(
    #         single_bench_df
    #     )
    #     title = "{} ({})".format(branch_name, bench_name)

    #     total_time_records = time_df.time.xs(
    #         [branch_name, bench_name], level=["commit_hash", "name"]
    #     )

    #     assert (
    #         len(total_time_records) == 1
    #     ), "there should be only 1 total time"

    #     total_time = total_time_records.iloc[0]
    #     extra_text = text = "total time: {:.3f}".format(total_time)

    #     plot_benchmark_dfs(df_mean, df_cumsum, title, extra_text)
