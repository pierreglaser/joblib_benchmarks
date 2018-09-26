#!/usr/bin/env/bash
set -o errexit
set -o nounset
set -o xtrace

# This script is expected to be run from the root of the joblib_benchmarks
# repository.
JOBLIB_BENCHMARKS_DIRECTORY=$PWD

# all repositories, python distributions and benchmaks will be ran here
BENCHMARKS_TEMP_DIR="/tmp/run_joblib_benchmarks"
mkdir "${BENCHMARKS_TEMP_DIR}"
cd "${BENCHMARKS_TEMP_DIR}"

# first, we install python using miniconda
# this version comes with python3.6
wget https://repo.continuum.io/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "${BENCHMARKS_TEMP_DIR}/miniconda"
export PATH="${BENCHMARKS_TEMP_DIR}/miniconda/bin:${PATH}"

echo "creating a python3.6 virtualenv..."
conda create -y -n joblib_benchmarks_virtualenv pip
source activate joblib_benchmarks_virtualenv

# echo "...done. Cloning scikit-learn..."
# this is not used for now in the benchmarks
# git clone https://github.com/scikit-learn/scikit-learn

echo "...done. Cloning joblib..."
git clone https://github.com/joblib/joblib

echo "...done. Copying joblib_benchmarks..."
cp -r "${JOBLIB_BENCHMARKS_DIRECTORY}" joblib_benchmarks

echo "...done. Installing asv..."
pip install asv

cd joblib_benchmarks

echo "done. Running the benchmarks..."
# specify informations about the machine we are using
asv machine --yes

# run all benchmarks for the last commit on joblib's master
asv run master^!
