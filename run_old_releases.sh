#!/usr/bin/env sh

# first get the tags from the joblib repository

if cd "$JOBLIB_PATH"; then
	echo "listing tags from joblib..."
else
	echo "JOBLIB_PATH is either not set or
	pointing to a wrong location, aborting operation."
fi

# we start bechmarking releases when joblib is written in
# python3, e.g approximately v0.8.0
OLDEST_BENCHMARKED_RELEASE="0.12.5"

# this lists the tags which history contains the
# $OLDEST_BENCHMARKED_RELEASE, e.g tags newer than
# $OLDEST_BENCHMARKED_RELEASE
ALL_TAGS=$(git tag --sort=committerdate --contains $OLDEST_BENCHMARKED_RELEASE)
echo "asv will benchmark the following tags: 
$ALL_TAGS"

# we now move to the joblib_benchmarks repository and start
# running the bechmarks
if cd "$JOBLIB_BENCHMARKS_PATH"; then 
	echo "now moving to joblib_benchmarks directory..."
else 
	echo "JOBLIB_BENCHMARKS_PATH is either not set or 
	pointing to a wrong location, aborting operation."
fi

for tag in $ALL_TAGS; do
	echo "running asv for tag $tag"
	# asv run "$tag"^! -b time_bytes_as_output
	asv run "$tag"^!
done
