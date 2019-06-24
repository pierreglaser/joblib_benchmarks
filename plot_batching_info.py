import pathlib

import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import git
from asv.config import Config

from asv_to_pandas import create_benchmark_dataframe
from benchmarks.bench_auto_batching import AutoBatchingSuite

params = {
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.size": 16,
}
mpl.rcParams.update(params)

asv_config = Config.load()
joblib_repo = git.Repo(asv_config.repo)
benchmark_suite_dir = pathlib.Path(__file__).parent
plots_path = pathlib.Path(benchmark_suite_dir) / "plots"

BENCHMARK_INFO_COLS = [
    "batch_idx",
    "batch_size",
    "total_duration",  # include the time spent in the queue
    "worker_duration",  # only the time spent calling the batch
    "smoothed_duration",
    "previous_smoothed_duration",
    "used_to_compute_bs",
    "last_influential_batch_idx",
    "compute_duration",  # total theroretical time needed to run the tasks
]

branches = [
    "master",
    "master-worker-duration",
    "master-slow-increase",
    "synchronize-batch-operations",
    "synchronized-slow-increase",
    "use-most-recently-sent-batch",
    "synchronize-logging-and-logic-ops",
    "use-time-spent-in-worker",
    "default-batching-strategy",
    "default-slow-increase",
    "default-slow-increase-fast-decrease",
    "most-recent-batch-slow-increase",
    "most-recent-batch-slow-increase-fast-decrease",
    "most-recent-batch-slow-increase-fast-decrease-end-to-end-duration",
]

hash_to_branch_map = {}
for h in joblib_repo.heads:
    if h.name in branches:
        hash_to_branch_map[h.commit.hexsha] = h.name

tags = [
    "use_smoothing",
    "increase_strategy",
    "decrease_strategy",
    "batch duration",
]

branch_tags_map = {
    "use-most-recently-sent-batch": [True, "fast", "slow", "end-to-end"],
    "synchronize-logging-and-logic-ops": [True, "fast", "slow", "end-to-end"],
    "use-time-spent-in-worker": [True, "fast", "slow", "compute time"],
    "default-batching-strategy": [True, "fast", "slow", "end-to-end"],
    "default-slow-increase": [True, "slow", "slow", "end-to-end"],
    "default-slow-increase-fast-decrease": [
        True,
        "slow",
        "fast",
        "end-to-end",
    ],  # noqa
    "most-recent-batch-slow-increase": [False, "slow", "slow", "compute-time"],
    "most-recent-batch-slow-increase-fast-decrease": [
        False,
        "slow",
        "fast",
        "compute-time",
    ],  # noqa
    "most-recent-batch-slow-increase-fast-decrease-end-to-end-duration": [
        False,
        "slow",
        "fast",
        "end-to-end",
    ],  # noqa
}

benchmark_names = [
    "track_cyclic_trend",
    "track_high_variance_no_trend",
    "track_low_variance_no_trend",
    "track_partially_cached",
]

bench_name_to_sequence_name_map = {
    "track_cyclic_trend": "cyclic",
    "track_high_variance_no_trend": "high_variance",
    "track_low_variance_no_trend": "low_variance",
    "track_partially_cached": "partially_cached",
}

BENCH_INST = AutoBatchingSuite()
BENCH_INST.setup(10000, 0.8, 4)  # parameters dont influence tasks running time


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


def reformat_single_benchmark_dataframe(df, bench_name):
    df = df.set_index("batch_idx", drop=False)
    assert not df.index.duplicated().sum(), "index contains duplicates!"
    df["task_idx"] = df.batch_size.cumsum()
    df.set_index("task_idx", inplace=True)

    # make batch size graphs more intuitive
    df = df.shift(-1)

    # include original tasks lengths
    df = df.reindex(range(1, df.index.max() + 1))
    df.batch_idx = df.batch_idx.ffill()

    tasks_times = getattr(
        BENCH_INST, bench_name_to_sequence_name_map[bench_name]
    )
    df["tasks_times"] = tasks_times

    df = df.ffill()

    df_mean = df.copy()
    # df_mean.total_duration /= df_mean.batch_size
    # df_mean.smoothed_duration /= df_mean.batch_size
    # df_mean.compute_duration /= df_mean.batch_size

    df_cumsum = df.cumsum()

    return df_mean, df_cumsum


def plot_benchmark_dfs(
    df_mean,
    df_cumsum=None,
    title=None,
    extra_text=None,
    include_cumulative_plots=True,
):
    if df_cumsum is not None:
        f, (ax_mean, ax_cumsum, ax_taskstimes) = plt.subplots(
            3, 1, gridspec_kw={"height_ratios": [3, 3, 1]}, figsize=(10, 12)
        )
        dfs = [df_mean, df_cumsum]
        axs = [ax_mean, ax_cumsum]
    else:
        f, (ax_mean, ax_taskstimes) = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
            figsize=(10, 12),
        )
        axs = [ax_mean]
        dfs = [df_mean]

    ax_taskstimes.plot(df_mean.index, df_mean.tasks_times, label="task times")
    ax_taskstimes.set_xlabel("task number")
    ax_taskstimes.set_ylabel("task time")

    cm = plt.get_cmap("tab10")
    colors = iter(cm.colors)

    for ax, df in zip(axs, dfs):
        ax.plot(
            df.index,
            df.total_duration,
            color=next(colors),
            label="total_duration",
        )
        ax.plot(
            df.index,
            df.compute_duration,
            color=next(colors),
            label="compute_duration",
        )
        ax.plot(
            df.index,
            df.previous_smoothed_duration,
            color=next(colors),
            label="previous_smoothed_duration",
        )

        twinax = ax.twinx()
        twinax.plot(
            df.index, df.batch_size, color=next(colors), label="batch_size"
        )

        handles, labels = ax.get_legend_handles_labels()
        tw_handles, tw_labels = twinax.get_legend_handles_labels()
        ax.legend(
            handles + tw_handles,
            labels + tw_labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0, 1, 1, 0.2),
        )

        ax.set_ylabel("task time")
        twinax.set_ylabel("batch size")

        # ax.set_title(title)

        # place text outside of the plot
        ax.text(
            0.05,
            1.25,
            extra_text,
            transform=ax.transAxes,
            bbox=dict(alpha=0.2),
            # horizontalalignment='center'
        )

    # make room for the text
    f.subplots_adjust(top=0.75)
    return f


def compare(bench_name, branch_one, branch_two):
    batch_df, time_df = aggregate_benchmark_dataframes()
    batch_df_dev = batch_df.xs(
        [branch_one, bench_name],
        level=["commit_hash", "name"],
        drop_level=False,
    )

    batch_df_master = batch_df.xs(
        [branch_two, bench_name],
        level=["commit_hash", "name"],
        drop_level=False,
    )
    return batch_df_dev, batch_df_master


def _format_figure_text(time, branch_name):
    total_time_str = f"Total time: {total_time:.3f}"
    extra_text = f"{total_time_str:^83}\n\n"

    branch_tags = branch_tags_map[branch_name]
    for i, (tag_name, tag_val) in enumerate(zip(tags, branch_tags)):
        if i % 2 == 0:
            new_field = f"{tag_name}:{tag_val}"
            new_field = f"{new_field:<30}"
        else:
            new_field = f"{tag_name:}:{tag_val}\n"
            new_field = f"{new_field:>30}"
        extra_text += new_field
    return extra_text


if __name__ == "__main__":
    batch_df, time_df = aggregate_benchmark_dataframes()

    batch_df = batch_df.xs(
        "default-batching-strategy", level="commit_hash", drop_level=False
    )

    # batch_df_dev, batch_df_master = compare(
    #     # "track_cyclic_trend",
    #     "track_partially_cached",
    #     # "synchronized-slow-increase",
    #     "synchronize-batch-operations",
    #     # "master-slow-increase",
    #     "master",
    # )

    by_batch_branch_and_benchmark = batch_df.groupby(
        ["commit_hash", "name"], axis=0
    )

    for name, single_bench_df in by_batch_branch_and_benchmark:
        branch_name, bench_name = name
        if bench_name != "track_low_variance_no_trend":
            continue
        df_mean, df_cumsum = reformat_single_benchmark_dataframe(
            single_bench_df, bench_name=bench_name
        )
        title = "{}\n(branch: {})".format(bench_name, branch_name)

        total_time_records = time_df.time.xs(
            [branch_name, bench_name], level=["commit_hash", "name"]
        )

        assert (
            len(total_time_records) == 1
        ), "there should be only 1 total time"

        total_time = total_time_records.iloc[0]
        extra_text = _format_figure_text(total_time, branch_name)
        f = plot_benchmark_dfs(
            df_mean,
            # df_cumsum,
            None,
            title,
            extra_text,
        )
        plot_dir = plots_path / bench_name
        plot_dir.mkdir(exist_ok=True)
        f.savefig(plot_dir / f"{branch_name}.png", dpi=f.dpi)
