# branches='use-most-recently-sent-batch synchronize-logging-and-logic-ops use-time-spent-in-worker default-batching-strategy default-slow-increase default-slow-increase-fast-decrease most-recent-batch-slow-increase most-recent-batch-slow-increase-fast-decrease most-recent-batch-slow-increase-fast-decrease-end-to-end-duration'

# for branch in $branches; do
# 	asv run -e -b bench_auto_batching.AutoBatchingSuite.track_ "${branch}"^!
# done

# for branch in default-slow-increase most-recent-batch-slow-increase; do
# 	asv run -e -b bench_auto_batching.AutoBatchingSuite.track_c ${branch}^!
# done
# for branch in most-recent-batch-slow-increase-fast-decrease; do
# 	asv run -e -b bench_auto_batching.AutoBatchingSuite.track_c ${branch}^!
# done

# for branch in most-recent-batch-slow-increase-fast-decrease-end-to-end-duration; do
# 	asv run -e -b bench_auto_batching.AutoBatchingSuite.track_c ${branch}^!
# done

for branch in default-batching-strategy; do
	asv run -ev -b bench_auto_batching.AutoBatchingSuite.track_l ${branch}^!
done
