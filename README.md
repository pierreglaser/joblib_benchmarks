Repository containing the speed benchmarking results and config files for joblib.
using [airspeed velocity](https://asv.readthedocs.io/en/stable/).
To get started, 

* make sure you have python 3.5 installed, as well as pip
* depending on your needs, you can use either virtualenv or conda to create environments. 
Simply update the `environment_type` field in `asv.conf.json`
* move to a directory where you have write access (for example your `$HOME`): `cd path/to/folder`
* clone this repository:
`git clone https://github.com/pierreglaser/joblib_benchmarks.git` 
* export the ASV_PYTHONPATH environment variable so that joblib can pickle objects belonging to the `benchmarks.py`file: 
`export ASV_PYTHONPATH=path/to/folder/joblib_benchmarks`

