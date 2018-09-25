#!/usr/bin/env/bash
set -o errexit
set -o nounset
set -o xtrace

# we assume that the script is ran from within the joblib_benchmarks
# repository, so that there exists a launch script doing:
# "git clone joblib_bencmarks && cd joblib_benchmarks"
# "source remote_run_script.sh"

# all repositories are created in the parent directory of joblib_benchmarks
cd "${HOME}"

# first, we install python using miniconda
# this version comes with python3.6
wget https://repo.continuum.io/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "${HOME}/miniconda"
export PATH="${HOME}/miniconda/bin:${PATH}"

mkdir temp_folder
cd temp_folder

echo "creating a python3.6 virtualenv..."
conda create -y -n joblib_benchmarks_virtualenv pip
source activate joblib_benchmarks_virtualenv

# echo "...done. Cloning scikit-learn..."
# this is not used for now in the benchmarks
# git clone https://github.com/scikit-learn/scikit-learn

echo "...done. Cloning joblib..."
git clone https://github.com/joblib/joblib

echo "...done. Copying joblib_benchmarks..."
cp -r "${HOME}/joblib_benchmarks" joblib_benchmarks

echo "...done. Installing asv..."
pip install asv

cd joblib_benchmarks

echo "done. Running the benchmarks..."
# specify informations about the machine we are using
asv machine --yes

# run all benchmarks for the last commit on joblib's master
asv run master^!
