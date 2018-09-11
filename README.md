Repository containing the speed benchmarking results and config files for joblib.
using [airspeed velocity](https://asv.readthedocs.io/en/stable/).
To get started, 

* make sure you have python 3.6 installed, as well as pip
* install asv and virtualenv: `python3.6 -m pip install asv virtualenv`
* move to a directory where you have write access (for example your home): `cd path/to/folder`
* clone the head fork of joblib: `git clone https://github.com/joblib/joblib.git`
* clone this repository `git clone https://github.com/pierreglaser/joblib_benchmarks.git 
* export the ASV_PYTHONPATH environment variable so that joblib can pickle objects belonging to the `benchmarks.py`file: 
`export ASV_PYTHONPATH=path/to/folder/joblib_benchmarks`

