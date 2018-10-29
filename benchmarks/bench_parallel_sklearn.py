import os
import timeit

import numpy as np

from benchmarks.common import EstimatorWithLargeList


class SklearnBenchmark:
    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0
    timer = timeit.default_timer
    timeout = 120

    # non-asv class attributes
    n_tasks = 10

    def setup(self, backend, pickler):
        # tell scikit-learn where to look for joblib
        os.environ['SKLEARN_SITE_JOBLIB'] = os.path.join(
                os.environ['ASV_ENV_DIR'], 'project')
        os.environ['LOKY_PICKLER'] = pickler


class TwentyDataBench(SklearnBenchmark):
    param_names = ['backend', 'pickler', 'n_jobs']
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['', 'cloudpickle'],
              [1, 2, 4])

    def setup(self, backend, pickler, n_jobs):
        super(TwentyDataBench, self).setup(backend, pickler)

        from sklearn.datasets import fetch_20newsgroups
        self.twenty_data = fetch_20newsgroups()

    def time_text_vectorizer(self, backend, pickler, n_jobs):
        from sklearn.linear_model.stochastic_gradient import SGDClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import ShuffleSplit, cross_val_score

        from joblib import parallel_backend

        pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                             ('clf', SGDClassifier())])

        cv = ShuffleSplit(n_splits=4, test_size=0.33)
        if n_jobs > 1:
            with parallel_backend(backend=backend):
                cross_val_score(pipeline, self.twenty_data.data,
                                self.twenty_data.target, cv=cv,
                                n_jobs=n_jobs)
        else:
            cross_val_score(pipeline, self.twenty_data.data,
                            self.twenty_data.target, cv=cv,
                            n_jobs=n_jobs)


class CaliforniaHousingBench(SklearnBenchmark):
    param_names = ['backend', 'pickler', 'n_jobs']
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['', 'cloudpickle'],
              [1, 2, 4])

    def setup(self, backend, pickler, n_jobs):
        super(CaliforniaHousingBench, self).setup(backend, pickler)

        from sklearn.datasets import fetch_california_housing
        self.california_data = fetch_california_housing()

    def time_kbins_polynomial_pipeline(self, backend, pickler, n_jobs):
        from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import ShuffleSplit, cross_val_score
        from joblib import parallel_backend

        pipeline = Pipeline([
            ('discretizer', KBinsDiscretizer(encode='onehot')),
            ('polynomial_features', PolynomialFeatures()),
            ('estimator', Ridge())])
        cv = ShuffleSplit(n_splits=4, test_size=0.3)

        if n_jobs > 1:
            with parallel_backend(backend=backend):
                cross_val_score(pipeline, self.california_data.data,
                                self.california_data.target, cv=cv,
                                n_jobs=n_jobs)
        else:
            cross_val_score(pipeline, self.california_data.data,
                            self.california_data.target, cv=cv,
                            n_jobs=n_jobs)


class MakeRegressionDataBench(SklearnBenchmark):
    param_names = ['backend', 'pickler', 'n_jobs', 'n_samples', 'n_features']
    params = (['multiprocessing', 'loky', 'threading'][1:],
              ['', 'cloudpickle'],
              [1, 2, 4],
              [10000, 30000],
              [10])

    def setup(self, backend, pickler, n_jobs, n_samples, n_features):
        super(MakeRegressionDataBench, self).setup(backend, pickler)
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples, n_features)
        self.X = X
        self.y = y

    def time_send_list(self, backend, pickler, n_jobs, n_samples, n_features):
        from joblib import delayed, parallel_backend, Parallel

        if n_jobs > 1:
            with parallel_backend(backend=backend):
                Parallel(n_jobs=n_jobs)(delayed(lambda x: x)(
                    list(range(100000))) for _ in range(self.n_tasks))
        else:
            Parallel(n_jobs=n_jobs)(delayed(lambda x: x)(
                list(range(100000))) for _ in range(self.n_tasks))

    def time_gridsearch_large_list(self, backend, pickler, n_jobs, n_samples,
                                   n_features):

        from joblib import Parallel, parallel_backend

        # importing parallel_backend from externals will fail
        # from sklearn.externals.joblib import Parallel, parallel_backend

        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV

        # serializing object with big lists slow down cloudpickle-based
        # Picklers. If the benchmarks do not fulfill these expectations,
        # something wrong is going on.
        r = EstimatorWithLargeList()
        params = {'alpha': [1, 0.1, 0.001]}

        if n_jobs > 1:
            with parallel_backend(backend):
                g = GridSearchCV(r, params, cv=4, n_jobs=n_jobs)
                g.fit(self.X, self.y)
        else:
            g = GridSearchCV(r, params, cv=4, n_jobs=n_jobs)
            g.fit(self.X, self.y)

    def time_ridge_gridsearch(self, backend, pickler, n_jobs, n_samples,
                              n_features):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        from joblib import parallel_backend

        params = {'alpha': [2**-i for i in range(1, 40)]}
        ridge = Ridge()
        # Use a large cv value because ridge is very fast
        if n_jobs > 1:
            with parallel_backend(backend=backend):
                rcv = GridSearchCV(ridge, params, cv=50, n_jobs=n_jobs)
                rcv.fit(self.X, self.y)
        else:
            rcv = GridSearchCV(ridge, params, cv=50, n_jobs=n_jobs)
            rcv.fit(self.X, self.y)
    time_ridge_gridsearch.param_names = [
            'backend', 'pickler', 'n_jobs', 'n_samples', 'n_features']
    time_ridge_gridsearch.params = (
            ['multiprocessing', 'loky', 'threading'][1:],
            ['', 'cloudpickle'],
            [1, 2, 4],
            # ridge is very fast, so use larger datasets
            [10000, 30000],
            [10])

    def time_randomforest(self, backend, pickler, n_jobs, n_samples,
                          n_features):
        from sklearn.ensemble.forest import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        from joblib import parallel_backend

        if n_jobs > 1:
            with parallel_backend(backend):
                rf = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs)
                rf.fit(self.X, self.y)
        else:
            rf = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs)
            rf.fit(self.X, self.y)

    def time_scaler_kernelridge_pipeline(self, backend, pickle, n_jobs,
                                         n_samples, n_features):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import ShuffleSplit, cross_val_score

        from joblib import parallel_backend

        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('estimator', KernelRidge())])

        cv = ShuffleSplit(n_splits=8, test_size=0.3)

        if n_jobs > 1:
            with parallel_backend(backend=backend):
                cross_val_score(pipeline, self.X, self.y, cv=cv, n_jobs=n_jobs)
        else:
            cross_val_score(pipeline, self.X, self.y, cv=cv, n_jobs=n_jobs)
    time_scaler_kernelridge_pipeline.param_names = [
            'backend', 'pickler', 'n_jobs', 'n_samples', 'n_features']
    time_scaler_kernelridge_pipeline.params = (
            ['multiprocessing', 'loky', 'threading'][1:],
            ['', 'cloudpickle'],
            [1, 2, 4],
            [10000],
            [10])
