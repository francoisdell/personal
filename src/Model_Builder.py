__author__ = 'Andrew Mackenzie'
import os
import sys

if os.path.abspath(os.pardir) not in sys.path:
    sys.path.append(os.path.abspath(os.pardir))

# from PyInstaller import compat
# mkldir = join(compat.base_prefix, "Library", "bin")
# binaries = [(join(mkldir, mkl), '') for mkl in os.listdir(mkldir) if mkl.startswith('mkl_')]
# if not mkldir in os.environ['PATH']:
#         os.environ['PATH'] = mkldir + os.pathsep + os.environ['PATH']
# for b in binaries:
#     if not b[0] in os.environ['PATH']:
#         os.environ['PATH'] = b[0] + os.pathsep + os.environ['PATH']

from typing import Union
from math import exp
import math
from os.path import join
from Settings import Settings
from collections import OrderedDict
import numpy as np
import pprint
from prettytable import PrettyTable
import importlib
import platform
import pandas as pd
from datetime import datetime as dt
import pickle
import bisect
import Function_Toolbox as ft
from scipy import sparse
from sklearn import exceptions
import sklearn.feature_extraction.text as sk_text
import sklearn.preprocessing as sk_prep
from sklearn import feature_extraction as sk_feat
from sklearn import feature_selection as sk_feat_sel
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, ConstantKernel, WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, MultiTaskLasso, OrthogonalMatchingPursuit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import LinearRegression, RidgeClassifier, BayesianRidge, Lars, LassoLars
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomTreesEmbedding
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC, SVR, NuSVR, NuSVC
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost
from tensorflow.contrib.estimator.python.estimator.rnn import RNNClassifier, RNNEstimator
from tensorflow.contrib.estimator import regression_head
from numbers import Number
from sklearn import metrics
from statistics import mean
import shutil
import warnings

pd.options.mode.chained_assignment = None
# pd.set_option('display.height', 1000)
pd.set_option('display.width', shutil.get_terminal_size()[0])
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_colwidth', 100)
warnings.filterwarnings(action="ignore")
pandas_objects = (pd.DataFrame, pd.Series, pd.SparseDataFrame, pd.SparseSeries)
pandas_series_objs = (pd.Series, pd.SparseSeries)
pandas_dataframe_objs = (pd.DataFrame, pd.SparseDataFrame)


class ModelSet:
    def __init__(self,
                 final_models: Union[str, list],
                 initial_models: Union[str, list] = list(),
                 # model_loading: str=None  # options: None, 'init', 'all'
                 ):

        if isinstance(final_models, str):
            final_models = [final_models]
        if isinstance(initial_models, str):
            initial_models = [initial_models]

        self.initial_models = [Model(v, 'stack') for v in initial_models]
        self.final_models = [Model(v, 'final') for v in final_models]
        # self.model_loading = model_loading

    def get_models(self):
        return self.initial_models + self.final_models

    # def get_trained_models(self):
    #     model_list = [(v, 'stack') for v in self.initial_models]
    #     model_list.extend([(v, 'final') for v in self.final_models])
    #     return model_list

    def set_models(self, new_models):
        for old_model in self.final_models:
            for new_model in new_models.final_models:
                if new_model.get_info() == old_model.get_info():
                    old_model.trained_model = new_model.trained_model
                    old_model.grid_param_dict = new_model.grid_param_dict
                    break

    def __str__(self):
        return 'modelset_{0}stacked'.format(len(self.initial_models))


class Model:
    def __init__(self,
                 model_class: str,
                 model_usage: str,
                 trained_model: object = None,
                 grid_param_dict: dict = None,
                 x_columns: list = None):
        self.model_class = model_class
        self.model_usage = model_usage
        self.trained_model = trained_model
        self.grid_param_dict = grid_param_dict
        self.is_custom = not isinstance(model_class, str)

    def get_info(self):
        return self.model_class, self.model_usage

    def __str__(self):
        return self.model_class + '_' + self.model_usage


class ModelBuilder:
    def __init__(self,
                 df: pd.DataFrame,
                 x_fields: dict,
                 y_field: Union[dict, OrderedDict],
                 model_type: Union[str, list, ModelSet],
                 report_name: str,
                 show_model_tests: bool = False,
                 retrain_model: bool = False,
                 selection_limit: float = 5.0e-2,
                 predict_all: bool = False,
                 verbose: bool = True,
                 train_pct: float = 0.7,
                 random_train_test: bool = True,
                 max_train_data_points: int = 50000,
                 pca_explained_var: float = 1.0,
                 stack_include_preds: bool = True,
                 final_include_data: bool = True,
                 cross_val_iters: tuple = (5, 3),
                 cross_val_model=None,
                 use_test_set: bool = True,
                 auto_reg_periods: int = 0,
                 correlation_method: str = 'corr',
                 correlation_max: float = 0.95,
                 correlation_significance_preference=True,
                 use_sparse: bool = True,
                 codify_nulls: bool = True,
                 allow_multiprocessing: bool=True
                 ):


        # df_idx_u = df.index.unique()
        # print(len(df_idx_u))
        # print(df[df.index.duplicated()])
        if df.index.duplicated().any():
            # lol = df.index.duplicated()
            # derp = df.index.values
            # print(*df.index.duplicated(), sep='\n')
            df.reset_index(drop=True, inplace=True)

        self.df = df
        self.x_fields = x_fields
        self.y_field = y_field
        self.model_type = model_type
        self.report_name = report_name
        self.show_model_tests = show_model_tests
        self.retrain_model = retrain_model
        self.selection_limit = selection_limit
        self.predict_all = predict_all
        self.verbose = verbose
        self.train_pct = train_pct
        self.random_train_test = random_train_test
        self.max_train_data_points = max_train_data_points
        self.pca_explained_var = pca_explained_var
        self.stack_include_preds = stack_include_preds
        self.final_include_data = final_include_data
        self.cross_val_iters = cross_val_iters
        self.cross_val_model = cross_val_model
        self.use_sparse = use_sparse
        self.s = Settings(report_name=report_name)
        self.use_test_set = use_test_set
        self.auto_reg_periods = auto_reg_periods
        self.correlation_method = correlation_method
        self.correlation_max = correlation_max
        self.codify_nulls = codify_nulls
        self.allow_multiprocessing = allow_multiprocessing

    def predict(self) -> (pd.DataFrame, str):

        s = Settings(report_name=self.report_name)
        data_file_dir = s.get_report_data_dir()

        if not self.verbose:
            # warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=exceptions.ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        if isinstance(self.model_type, list):
            self.model_type = ModelSet(self.model_type[-1], self.model_type[:-1])
        elif isinstance(self.model_type, str):
            self.model_type = ModelSet(self.model_type)

        model_dir = s.get_model_dir()

        predictive_model_file = '{0}/{1}'.format(model_dir, 'pred_models.p')
        mappings_file = '{0}/{1}'.format(model_dir, 'data_mappings.p')

        for key in list(self.y_field.keys()):
            self.df[key].replace('', np.nan, inplace=True)
            # print("Value Counts in dataframe\n", df[k].value_counts())
            if self.predict_all:
                mask_validate = np.asarray([True for v in range(self.df.shape[0])], dtype=np.bool)
            else:
                mask_validate = self.df[key].isnull()
            mask_train_test = ~self.df[key].isnull()
            # print('Final Status field\n', df[k])
            # print('Value Counts\n', mask_train_test.value_counts())
            # print('DF Length\n', len(df[k]))
            # print('Validate Dataset\n', mask_train_test.value_counts().loc[True])

            mask_train_test_qty_true = mask_train_test.value_counts().loc[True]

            self.train_pct = min(self.train_pct,
                                 self.max_train_data_points / (len(self.df[key]) - mask_train_test_qty_true))
            test_pct = min((1 - self.train_pct),
                           self.max_train_data_points / (len(self.df[key]) - mask_train_test_qty_true))

            if self.verbose:
                print("Train data fraction:", self.train_pct)
                print("Test data fraction:", test_pct)

            if self.random_train_test:
                mask = np.random.rand(len(self.df[key]))
                mask_train = [bool(a and b) for a, b in zip(mask <= self.train_pct, mask_train_test != False)]
                mask_test = [bool(a and b) for a, b in zip(mask > (1 - test_pct), mask_train_test != False)]
            else:
                mask_train = mask_train_test.copy()
                mask_test = mask_train_test.copy()
                mask_train_qty = 0
                for idx, val in enumerate(mask_train_test):
                    if val:
                        if float(mask_train_qty) <= mask_train_test_qty_true * self.train_pct:
                            mask_test[idx] = False
                        else:
                            mask_train[idx] = False
                        mask_train_qty += 1

            if self.use_test_set is False:
                print("Don't need to split the set into train/validate since we're using cross validation.")
                mask_train = mask_train_test

            # print("Mask_Train Value Counts\n", pd.Series(mask_train).value_counts())
            # print("Mask_Test Value Counts\n", pd.Series(mask_test).value_counts())
        # x_mappings, x_columns, x_vectors = set_vectors(df.loc[mask_train_test,], x_fields)
        # y_mappings, y_vectors = set_vectors(df, y_field, is_y=True)
        # print_sp500(self.df)
        try:
            if self.retrain_model:
                raise ValueError(
                    '"retrain_model" variable is set to True. Training new preprocessing & predictive models.')
            with open(mappings_file, 'rb') as pickle_file:
                x_mappings, y_mappings = pickle.load(pickle_file)
            with open(predictive_model_file, 'rb') as pickle_file:
                loaded_models, = pickle.load(pickle_file)

                # IF THERE ARE ANY PREEXISTING MODELS, USE THEM RATHER THAN RETRAINING THEM UNNECESSARILY
                # model_compare = [m1.model_usage == m2.model_usage and
                #                  m1.x_columns == m2.x_columns and
                #                  m1.model_class == m2.model_class and
                #                  np.testing.assert_equal(m1.grid_param_dict, m2.grid_param_dict)
                #                  for m1, m2 in zip(model_type.initial_models, loaded_models.initial_models)]
                # if model_compare:
                # if all(model_compare):
                #     model_type.set_final_models(loaded_models)

            print('Successfully loaded the mappings and predictive model.')
        except (FileNotFoundError, ValueError) as e:
            print(e)
        else:
            self.retrain_model = True

        # If autoregression has been requested, add the autoregression varibales
        # if self.auto_reg_periods > 0:
        #     for y_field, y_type in self.y_field.items():
        #         for lag in range(self.auto_reg_periods):
        #             y_field_lag_name = y_field + '_lag{0}'.format(lag+1)
        #             self.df[y_field_lag_name] = self.df[y_field].shift(lag-1)
        #             if np.isnan(self.df[y_field_lag_namee].iloc[0] = self.df[y_field_lag_name].mean()
        #             self.x_fields[y_field_lag_n].iloc[0]):
        #                 self.df[y_field_lag_namame] = y_type

        # CHANGE THE 'cat' to a list of all possible values in the y field. This is necessary because the LabelEncoder
        # that will encode the Y variable values can't handle never-before-seen values. So we need to pass it every
        # possible value in the Y variable, regardless of whether it appears in the train or test subsets.
        for k, v in self.y_field.items():
            if v == 'cat':
                self.y_field[k] = self.df[k].unique().astype(str)

        ################################################################################################################
        # If sparsity is requested, change the DataFrame to a SparseDataFrame
        ################################################################################################################
        if self.use_sparse:
            self.df = self.df.to_sparse()

        # print_sp500(self.df)

        y_mappings = self.train_models(self.df.loc[mask_train, :], self.y_field)
        _, y_train, y_mappings = self.get_vectors(self.df.loc[mask_train, :], self.y_field, y_mappings, is_y=True)
        y_train_names = y_train.columns.tolist()
        if self.df[self.df.index.duplicated()].shape[0] > 0:
            raise ValueError('Index of dataframe "df" contains dupes!!!')
        if y_train[y_train.index.duplicated()].shape[0] > 0:
            raise ValueError('Index of dataframe "y_train" contains dupes!!!')

        # print(ft.str_as_header('DATA CHECKING PROCESSES'))
        # print('Duplicates in self.df index (SHOULD BE 0): ', self.df[self.df.index.duplicated()].shape[0])
        # print('Duplicates in y_train index (SHOULD BE 0): ', y_train[y_train.index.duplicated()])

        update_df_outer(self.df, y_train)
        # update_df_outer(self.df, y_train)
        # self.df = self.df.combine_first(y_train)
        # print_sp500(self.df)

        x_columns, x_mappings = self.get_fields(y_column_names=y_train_names, mask=mask_train)

        if self.verbose:
            print('FIELDS\n', np.asarray(list(self.x_fields.keys())))

        pred_x_fields = OrderedDict()
        pred_x_mappings = OrderedDict()
        pred_x_columns = OrderedDict()

        ########################################################################################################
        # For now we only support 1 y field. In the future we may want to be able to iterate over multiple y-fields
        #   and churn out predictions for each one. But that's a future problem.
        k, v = list(self.y_field.items())[0]
        ########################################################################################################

        ####################################################
        # INITIAL MODEL TRAINING AND PREDICTION GENERATION #
        ####################################################
        for idx, model in enumerate(self.model_type.initial_models):

            model = self.train_predictive_model(model,
                                                list(x_columns.keys()),
                                                self.df.loc[mask_train, list(x_columns.keys())],
                                                self.df.loc[mask_train, y_train_names])

            clf = model.trained_model

            # Print Model Results and save to a file
            model_outputs_dir = join(data_file_dir, f'{model.model_usage}_{model.model_class}')
            os.makedirs(model_outputs_dir, exist_ok=True)
            print_model_results(clf,
                                self.df.loc[mask_train, list(x_columns.keys())],
                                self.df.loc[mask_train, y_train_names],
                                model_outputs_dir, self.selection_limit)

            max_slice_size = 100000
            y_all = list()
            mask_all = np.asarray([True for v in range(self.df.shape[0])], dtype=np.bool)
            # y_all_probs = list()
            for s in range(0, int(math.ceil(len(self.df.index) / max_slice_size))):
                min_idx = s * max_slice_size
                max_idx = min(len(self.df.index), (s + 1) * max_slice_size)
                print(f"Prediction Iteration #{s}: min/max = {min_idx}/{max_idx}")

                df_valid_iter = self.df.loc[mask_all, :].iloc[min_idx:max_idx]
                x_valid_columns, x_all, x_mappings = self.get_vectors(df_valid_iter.loc[:, list(self.x_fields.keys())], self.x_fields, x_mappings)
                update_df_outer(self.df, x_all)
                # print_sp500(self.df)
                # print(x_validate)
                calculate_probs = hasattr(clf, 'classes_') \
                                  and hasattr(clf, 'predict_proba') \
                                  and not (hasattr(clf, 'loss') and clf.loss == 'hinge')

                preds, x_all = self.resilient_predict(clf, x_all.loc[:, list(x_columns.keys())])
                if calculate_probs:
                    pred_probs, x_all = self.resilient_predict_probs(clf, x_all)
                y_all.append(preds)

            # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
            y_all = np.hstack(y_all).reshape(-1, 1)

            # Generate predictions for this predictive model, for all values. Add them to the DF so they can be
            #  used as predictor variables (e.g. "stacked" upon) later when the next predictive model is run.
            # y_all = np.atleast_1d(y_all)
            if isinstance(y_mappings[k], sk_prep.LabelEncoder):
                y_all = y_all.astype(int)
            preds = y_mappings[k].inverse_transform(y_all)
            print('INTERMEDIATE PREDICTIONS')
            print("First 7: ", *preds[:10])
            print("Last 7: ", *preds[-10:])

            # print_sp500(self.df)

            new_x_fields = {}
            if calculate_probs:
                pred_field_names = [k + '_' + model.model_class + '_prob_' + f.lower().replace(' ', '_') for f in
                                    sorted(list(y_mappings[k].classes_))]  #[:pred_probs.shape[1]]]
                pred_field_names = [get_unique_name(v, self.df.columns.values) for v in pred_field_names]
                df_pred = pd.DataFrame(data=pred_probs
                                       , columns=pred_field_names
                                       , index=df_valid_iter.index)
                df_pred = df_pred.loc[:, df_pred.columns.values[1:]]
                for f in df_pred.columns.values:
                    new_x_fields[f] = 'num'
                # self.df = self.df.join(df_preds, how='inner')
                print('Unique probabilities: ', len(np.unique(pred_probs)))
                print('Probability variance: ', np.var(pred_probs))

            else:
                pred_field_names = get_unique_name('{0}_pred_{1}'.format(k, model.model_class),
                                                  self.df.columns.values)
                df_pred = pd.DataFrame(data=preds
                                       , columns=[pred_field_names]
                                       , index=df_valid_iter.index)

                new_x_fields[pred_field_names] = 'cat' if not isinstance(v, str) else v
                print('Unique predictions: ', len(np.unique(preds)))

            update_df_outer(self.df, df_pred)

            new_x_mappings = self.train_models(self.df.loc[mask_train, list(new_x_fields.keys())], new_x_fields)
            new_x_columns, new_x_column_df, new_x_mappings = \
                self.get_vectors(self.df.loc[mask_train, new_x_fields], new_x_fields, new_x_mappings)
            update_df_outer(self.df, new_x_column_df)

            if self.stack_include_preds:

                self.x_fields.update(new_x_fields)
                x_mappings.update(new_x_mappings)
                x_columns.update(new_x_columns)
                x_columns = self.reduce_dimensions(x_columns, k, mask_train)

            # Save the predictions for later. They'll be added into the dataset for the final model predictions
            pred_x_fields.update(new_x_fields)
            pred_x_mappings.update(new_x_mappings)
            pred_x_columns.update(new_x_columns)

            if hasattr(clf, 'intercept_') and (isinstance(clf.intercept_, (list)) and len(clf.intercept_) == 1):
                print('--- Model over-normalization testing ---\n'
                      'Intercept/Expit/Exp = {0} / {1} / {2}'
                      .format(format(clf.intercept_[0], '.4f')
                              , format(ft.sigmoid(clf.intercept_[0]), '.4f')
                              , format(exp(clf.intercept_[0])), '.4f'))

        ##################################################
        # FINAL MODEL TRAINING AND PREDICTION GENERATION #
        ##################################################
        for mdl in self.model_type.final_models:

            if self.model_type.initial_models:

                if not self.final_include_data:
                    self.x_fields = pred_x_fields
                    x_mappings = pred_x_mappings
                    x_columns = pred_x_columns
                elif not self.stack_include_preds:
                    self.x_fields.update(pred_x_fields)
                    x_mappings.update(pred_x_mappings)
                    x_columns.update(pred_x_columns)

                x_columns = self.reduce_dimensions(x_columns, k, mask_train)

            mdl = self.train_predictive_model(mdl,
                                              list(x_columns.keys()),
                                              self.df.loc[mask_train, list(x_columns.keys())],
                                              self.df.loc[mask_train, y_train_names])

            clf = mdl.trained_model
            # If PCA has been specified, convert x_fields to PCA
            # if self.pca_explained_var < 1:
            #     self.df, x_columns = self.convert_to_pca(pca_df=self.df,
            #                                              col_names=list(x_columns.keys()),
            #                                              explained_variance=self.pca_explained_var)

            model_outputs_dir = join(data_file_dir, f'{mdl.model_usage}_{mdl.model_class}')
            os.makedirs(model_outputs_dir, exist_ok=True)

            # INITIAL AND FINAL MODELS: SHOW MODEL TESTS
            if self.show_model_tests:
                print(ft.str_as_header('TEST SET APPLICATION'))
                _, x_test_df, x_mappings = self.get_vectors(self.df.loc[mask_test, list(self.x_fields.keys())],
                                                            self.x_fields, x_mappings)
                _, y_test_df, y_mappings = self.get_vectors(self.df.loc[mask_test, list(self.y_field.keys())],
                                                            self.y_field, y_mappings, is_y=True)
                update_df_outer(self.df, x_test_df)
                update_df_outer(self.df, y_test_df)

                print("Total Named Columns: {0}".format(len(x_columns)))

                # Print Model Results and save to a file
                print_model_results(clf, x_test_df.loc[:, list(x_columns.keys())], y_test_df,
                                    model_outputs_dir, self.selection_limit)

                if hasattr(clf, 'intercept_'):
                    intercept = clf.intercept_
                    if isinstance(clf.intercept_, (np.float64, float)):
                        intercept = [intercept]

                    print('INTERCEPT')
                    print(*intercept, sep='\n')

                    print('INTERCEPT (EXPIT)')
                    print(*[ft.sigmoid(v) for v in intercept], sep='\n')

                    print('INTERCEPT (EXP)')
                    print(*[ft.sigmoid(v) for v in intercept], sep='\n')

                preds, x_test_df = self.resilient_predict(clf, x_test_df.loc[:, list(x_columns.keys())])
                print('R2 SCORE\n', r2_score(y_test_df, preds))

                if hasattr(clf, 'classes_'):
                    print('CLASSES')
                    print(clf.classes_)
                    print('ACCURACY:\n', metrics.accuracy_score(y_test_df, preds))
                    try:
                        y_test_probs, x_test_df = self.resilient_predict_probs(clf, x_test_df.loc[:, list(x_columns.keys())])
                        if y_test_probs.shape[1] <= 2:
                            print('ROC-CURVE AOC', metrics.roc_auc_score(y_test_df, y_test_probs.loc[:, 1]))

                        print(ft.str_as_header('PREDICTION PROBABILITIES', char='-', end_chars=3))
                        print(f'Total Number of Classes: {y_test_probs.shape[1]}')
                        print(f'=== Max Probability of each class ===\n{y_test_probs.aggregate(max, axis=0)}')
                        print(f'=== Unique Probability Combinations ===\n{y_test_probs.nunique()}')
                        print(f'=== Sample Prediction Values ===\n{y_test_probs.head(10)}')
                    except (ValueError, AttributeError) as e:
                        print(e)
                        pass

                    for k, v in self.y_field.items():
                        y_test_preds = self.resilient_inverse_transform(y_mappings[k], preds)

                        print("Unique integers in y_test: {0}".format(y_test_df.iloc[:, 0].unique().tolist()))
                        y_test_actual = self.resilient_inverse_transform(y_mappings[k], y_test_df)

                        # Create confusion matrix
                        if y_test_actual.shape[1] != 1:
                            raise ValueError('Dataframe y_test_actual is not a 1d dataframe. Cannot use in a crosstab.')
                        if y_test_preds.shape[1] != 1:
                            raise ValueError('Dataframe y_test_preds is not a 1d dataframe. Cannot use in a crosstab.')
                        y_test_actual = y_test_actual.T.squeeze()
                        y_test_preds = y_test_preds.T.squeeze()
                        conf_matrix = pd.crosstab(y_test_actual.values, y_test_preds.values,
                                                  rownames=['actual'], colnames=['predicted'])
                        print("CONFUSION MATRIX")
                        print(conf_matrix)
                        conf_matrix.to_csv(join(model_outputs_dir, 'conf_matrix.csv'), header=True, encoding='utf_8')
                pass

            print("--------------------------------\n"
                  "-- VALIDATION SET APPLICATION --\n"
                  "--------------------------------")
            # CREATE X_VALIDATE
            df_validate = self.df.loc[mask_validate, :]

            # ITERATE OVER THE df_validate in groups of 100K rows (to avoid memory errors) and predict outcomes
            max_slice_size = 100000
            y_validate = pd.DataFrame()
            y_validate_probs = pd.DataFrame()
            # y_pred_name = 'pred_' + k
            for s in range(0, int(math.ceil(len(df_validate.index) / max_slice_size))):
                min_idx = s * max_slice_size
                max_idx = min(len(df_validate.index), (s + 1) * max_slice_size)
                print("Prediction Iteration #{0}: min/max = {1}/{2}".format(s, min_idx, max_idx))

                df_valid_iter = df_validate.iloc[min_idx:max_idx]
                _, x_validate, x_mappings = self.get_vectors(df_valid_iter, self.x_fields, x_mappings)
                # update_df_outer(self.df, x_validate)

                # print(x_validate)
                # print("Prediciton Matrix Shape: {0}".format(x_validate.shape))
                # if self.auto_reg_periods > 0:
                #     preds = []
                #     x_validate = []
                #     for r in range(x_validate.shape(0)):
                #         pred, x_validate_row = self.resilient_predict(clf, x_validate[r:r + 1, :])
                #         preds.append(preds)
                #         x_validate.append(x_validate_row)
                # else:
                preds, _ = self.resilient_predict(clf, x_validate.loc[:, list(x_columns.keys())])
                y_validate = y_validate.append(preds, verify_integrity=True)
                # update_df_outer(y_validate, preds)
                # update_df_outer(self.df, preds)

                calculate_probs = hasattr(clf, 'classes_') \
                                  and hasattr(clf, 'predict_proba') \
                                  and not (hasattr(clf, 'loss') and clf.loss == 'hinge')
                if calculate_probs:
                    pred_probs, _ = self.resilient_predict_probs(clf, x_validate.loc[:, list(x_columns.keys())])
                    y_validate_probs = y_validate_probs.append(pred_probs, verify_integrity=True)
                    # update_df_outer(y_validate_probs, pred_probs)
                    # update_df_outer(self.df, pred_probs)

            # print_sp500(self.df)

            # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
            # y_validate = np.hstack(y_validate).reshape(-1, 1).astype(int)

            self.new_fields = list()
            if calculate_probs:
                print('ORIGINAL PROBABILITIES')
                print(*y_validate_probs[:10].round(6), sep='\n')
                print('ORIGINAL PROBABILITIES (NORMALIZED)')
                print(*[np.around(np.expm1(x), 4) for x in y_validate_probs][:10], sep='\n')

                for y_name, mapping in y_mappings.items():
                    df_probs = pd.DataFrame(data=y_validate_probs.round(6),
                                            columns=[y_name + '_prob_' + f.lower().replace(' ', '_') for f in
                                                     list(mapping.classes_)[:y_validate_probs.shape[1]]],
                                            index=df_validate.index)

                    for name in df_probs.columns.values:
                        if name in df_validate.columns.values:
                            df_probs.rename(columns={name: name + '_final'}, inplace=True)
                    # df_validate = df_validate.join(df_probs, how='inner')
                    update_df_outer(df_validate, df_probs)
                self.new_fields.extend(df_probs.columns.values.tolist())

            # print_sp500(self.df)

            for k, v in self.y_field.items():
                if isinstance(y_mappings[k], sk_prep.LabelEncoder):
                    y_validate = y_validate.astype(int)
                else:
                    y_validate = y_validate.astype(float)
                final_preds = y_mappings[k].inverse_transform(y_validate)
                print('FINAL PREDICTIONS')
                print('First 7: ', *final_preds[:10])
                print('Last 7: ', *final_preds[-10:])
                y_pred_name = 'pred_' + k
                df_validate.loc[:, y_pred_name] = final_preds
                self.new_fields.extend([y_pred_name])
                # update_df_outer(self.df, df_validate)

            # if self.predict_all:
            #     return_df = df_validate
            # else:
            #     return_df = self.df.loc[mask_validate == 0, :].append(df_validate, ignore_index=True)

            print('===== Appending the validation data to the final dataframe =====')
            print("Shape Before:", self.df.shape)
            update_df_outer(self.df, df_validate)
            print("Shape After:", self.df.shape)
            if self.df is None:
                raise ValueError('DF is None!!1!')
            # print_sp500(self.df)

            yield mdl

        # INITIAL AND FINAL MODELS: TRAIN ANY UNTRAINED MODELS
        if self.verbose:
            print('FIELDS\n', np.asarray(list(self.x_fields.keys())))

        with open(predictive_model_file, 'wb') as pickle_file:
            pickle.dump((self.model_type,), pickle_file)
        with open(mappings_file, 'wb') as pickle_file:
            pickle.dump((x_mappings, y_mappings), pickle_file)

    def get_fields(self,
                   y_column_names: list,
                   mask: Union[list, np.ndarray, tuple]
                   ) -> (OrderedDict, OrderedDict):

        field_names = list(self.x_fields.keys())
        mappings = self.train_models(self.df.loc[mask, field_names], self.x_fields)
        x_columns, x, mappings = self.get_vectors(self.df.loc[mask, field_names], self.x_fields, mappings)
        update_df_outer(self.df, x)

        x_columns = self.reduce_dimensions(x_columns, y_column_names, mask)

        return x_columns, mappings

    def reduce_dimensions(self, x_columns: dict, y_column_names: list, mask: np.ndarray) -> dict:
        #####################################################
        # Remove Highly Correlated Variables (If Specified) #
        #####################################################
        if 0 < self.correlation_max < 1:
                if self.correlation_method == 'corr':
                    reduced_x_column_names = reduce_variance_corr(df=self.df.loc[mask, :],
                                                                  x_column_names=list(x_columns.keys()),
                                                                  y_column_names=y_column_names,
                                                                  max_corr_val=self.correlation_max
                                                                  )
                    [x_columns.pop(key) for key in list(x_columns.keys()) if key not in reduced_x_column_names]
                elif self.correlation_method == 'pca':
                    pca_df, reduced_x_column_names = reduce_variance_pca(df=self.df.loc[mask, :],
                                                                         column_names=list(x_columns.keys()),
                                                                         explained_variance=self.correlation_max
                                                                         )

                    x_columns = {k: None for k in reduced_x_column_names}
                else:
                    raise ValueError('Incorrect variance reduction method provided! [{0}]'.format(
                        self.correlation_method))

                print('X Names Length: {0}'.format(len(list(x_columns.keys()))))

        ###################################################
        # Remove Non-Significant Variables (If Specified) #
        ###################################################
        if self.selection_limit < 1.0:
            x_column_names = list(x_columns.keys())
            print('Pruning x_fields for any variables with a p-value > {0}'.format(self.selection_limit))
            if len(np.unique(self.df.loc[mask, y_column_names])) <= 1:
                raise ValueError('Not enough unique values in the target variable. Can\'t build model. Check your sh1t')
            elif len(np.unique(self.df.loc[mask, y_column_names])) == 2:
                scores, p_vals = sk_feat_sel.f_classif(self.df.loc[mask, x_column_names], self.df.loc[mask, y_column_names])
            else:
                scores, p_vals = sk_feat_sel.f_regression(self.df.loc[mask, x_column_names], self.df.loc[mask, y_column_names])

            # remaining_fields = set()
            for idx, col_name in ft.reverse_enumerate(x_column_names):
                if p_vals[idx] > self.selection_limit:
                    x_columns.pop(col_name)
                    print('Column [{0}] not found to be statistically significant. Removing.'.format(col_name))

        print(ft.str_as_header('After variable selection, there are {0} remaining variables'.format(len(x_columns))))
        return x_columns

    def get_vectors(self,
                    df: pd.DataFrame,
                    field_names: dict,
                    trained_models: dict,
                    is_y: bool = False
                    ) -> (dict,
                          Union[pandas_objects],
                          OrderedDict):

        df = df.loc[:, list(field_names.keys())]

        column_names = OrderedDict()
        for f, t in list(field_names.items()):
            if self.verbose:
                print("TRANSFORMING: ", f, end='')
            transformed_field_name = f + '_t'
            if self.codify_nulls and not is_y:
                null_field_name = f + '_isnull'
                # null_arr = df[f].isnull().astype(np.float64)
                # if len(np.unique(null_arr)) >= 2:  # un-comment to skip useless null fields. Will break the code...
                df[null_field_name] = df[f].isnull().astype(np.float64)
                column_names[null_field_name] = {'use': True, 'orig_name': f}

            # print(trained_models[f])
            if isinstance(trained_models[f], sk_prep.LabelEncoder):
                while True:
                    try:
                        df_vals = df[f].astype(np.str)
                        print('None values (MUST BE ZERO!): ', sum([x is None or x == np.NaN for x in df_vals]))

                        df[transformed_field_name] = pd.Series(data=trained_models[f].transform(df_vals), index=df_vals.index)
                        column_names[transformed_field_name] = {'use': True, 'orig_name': f}

                        break
                    except (ValueError, AttributeError) as e:
                        print("\nAll Uniques in Series:\n", sorted(df[f].unique().astype(str)))
                        print('Length: {}\n'.format(len(df[f].unique().astype(str))))
                        print("All Classes in Trained Models[{0}]{1}".format(f, sorted(trained_models[f].classes_.tolist())))
                        print('Length: {}\n'.format(len(trained_models[f].classes_.tolist())))
                        print(str(e))
                        new_classes = [v for k, v in enumerate(str(e).split("\'")) if k % 2 == 1]
                        le_classes = trained_models[f].classes_
                        if isinstance(le_classes, np.ndarray):
                            le_classes = le_classes.tolist()
                            for v in le_classes:
                                if isinstance(v, (list, tuple)):
                                    le_classes = v
                                    print('Extracted values from ndarray: ', le_classes)
                                    break
                        for v in new_classes:
                            bisect.insort_left(le_classes, v)
                        trained_models[f].classes_ = le_classes
                        # trained_models[f].classes_ = np.append(trained_models[f].classes_, new_classes)
                        print(f'Added new classes to encoder: {new_classes} ')
                        print(f'All Classes: {sorted(trained_models[f].classes_)}')
                        print(f'Length: {len(trained_models[f].classes_)}')
                        pass

            elif t == 'cat':
                mkdict = lambda row: dict((col, row[col]) for col in [f])
                matrix = trained_models[f].transform(df.apply(mkdict, axis=1))
                for col_index, col_name in enumerate(trained_models[f].get_feature_names()):
                    transformed_field_name = f + '_' + col_name
                    df.loc[:, transformed_field_name] = matrix[:, col_index].astype(np.float64)
                    column_names[transformed_field_name] = {'use': True, 'orig_name': f}

            else:
                if t == 'doc':
                    matrix = trained_models[f].transform(df[f].values.astype(str))
                    if isinstance(df, pd.DataFrame):
                        matrix = matrix.toarray()
                    for col_index, col_name in enumerate(trained_models[f].get_feature_names()):
                        adj_col_name = f + '_' + col_name.encode('ascii', errors='ignore').decode('utf-8', errors='ignore')
                        column_names[adj_col_name] = {'use': True, 'orig_name': f}
                        df[adj_col_name] = matrix[:, col_index].astype(np.float64)
                # else:
                elif t == 'num':
                    df[transformed_field_name] = trained_models[f].transform(df[f]
                                                                             .apply(pd.to_numeric, errors='coerce')
                                                                             .round(6).fillna(0).values.reshape(-1, 1))
                    if is_y:
                        df[transformed_field_name] = df[transformed_field_name].astype(np.float16)  # Orig was float64

                        if isinstance(trained_models[f], sk_prep.MinMaxScaler):
                            df[transformed_field_name] = np.minimum(1, np.maximum(0, fix_np_nan(df[transformed_field_name])))

                    else:

                        df[transformed_field_name] = df[transformed_field_name].astype(np.float64)
                        if isinstance(trained_models[f], sk_prep.MinMaxScaler):
                            df[transformed_field_name] = np.minimum(1, np.maximum(0, fix_np_nan(df[transformed_field_name])))
                        column_names[transformed_field_name] = {'use': True, 'orig_name': f}

                else:
                    raise ValueError('Type provided [(0}] for field [{1}] is not supported.'.format(t, f))

            if self.verbose:
                print(' || Shape: {0}'.format(df.shape))

        df.drop(list(field_names.keys()), inplace=True, axis=1)
        df = fix_np_nan(df)
        return column_names, df, trained_models

    def convert_to_pca(self,
                       pca_df: Union[pandas_dataframe_objs],
                       col_names: list,
                       explained_variance: float
                       ) -> (Union[pandas_dataframe_objs], Union[dict, OrderedDict]):

        print("Conducting PCA and pruning components above the desired explained variance ratio")
        max_components = len(col_names) - 1
        pca_model = TruncatedSVD(n_components=max_components, random_state=555)

        x_results = pca_model.fit_transform(pca_df.loc[:, col_names]).T
        print(pca_model.components_)
        print(pca_model.explained_variance_ratio_)

        x_names_pca = OrderedDict()
        sum_variance = 0
        for idx, var in enumerate(pca_model.explained_variance_ratio_):
            sum_variance += var
            pca_name = 'pca_{0}'.format(idx)
            pca_df[pca_name] = x_results[idx]
            x_names_pca[pca_name] = {'use': True, 'orig_name': None}
            if sum_variance > explained_variance:
                break
        return pca_df, x_names_pca

    def resilient_fit(self,
                      obj,
                      x: Union[pandas_dataframe_objs],
                      y: Union[pandas_series_objs]) \
            -> GridSearchCV:
        try:
            if isinstance(y, pandas_objects):
                y = y.squeeze()
                y = y.ravel()  # Turns the DF
            obj.fit(x, y)
        except (TypeError, ValueError, DeprecationWarning) as e:
            if 'Expected 2D' in str(e) and len(x.shape) == 1:
                # print(e)
                print('=== 1D Array Error Caught. Reshaping array to overcome the error. ===')
                x = x.reshape(-1, 1)
                return self.resilient_fit(obj, x, y)
            elif "dense" in str(e) and sparse.issparse(x):
                x = x.to_dense()
                self.use_sparse = False
                return self.resilient_fit(obj, x, y)
            elif hasattr(obj, 'n_jobs') and obj.n_jobs != 1:
                obj.n_jobs = 1
                return self.resilient_fit(obj, x, y)
            elif 'infs or NaNs' in str(e) and x.dtype == np.float64:
                x = x.astype(np.float32)
                return self.resilient_fit(obj, x, y)
            elif 'infs or NaNs' in str(e) and x.dtype == np.float32:
                x = x.astype(np.float16)
                return self.resilient_fit(obj, x, y)
            else:
                raise

        return obj

    def resilient_predict(self,
                          obj,
                          x: pd.DataFrame) \
            -> (pd.DataFrame, pd.DataFrame):
        try:
            preds = obj.predict(x)
        except (TypeError, ValueError, DeprecationWarning) as e:
            if 'Expected 2D' in str(e) and len(x.shape) == 1:
                # print(e)
                print('=== 1D Array Error Caught. Reshaping array to overcome the error. ===')
                x = x.reshape(-1, 1)
                return self.resilient_predict(obj, x)
            elif "dense" in str(e) and sparse.issparse(x):
                x = x.to_dense()
                self.use_sparse = False
                return self.resilient_predict(obj, x)
            elif hasattr(obj, 'n_jobs') and obj.n_jobs != 1:
                obj.n_jobs = 1
                return self.resilient_predict(obj, x)
            elif 'infs or NaNs' in str(e) and x.dtype == np.float64:
                x = x.astype(np.float32)
                return self.resilient_predict(obj, x)
            elif 'infs or NaNs' in str(e) and x.dtype == np.float32:
                x = x.astype(np.float16)
                return self.resilient_predict(obj, x)
            else:
                raise

        return pd.DataFrame(data=preds, index=x.index), x

    def resilient_predict_probs(self,
                                obj,
                                x: pd.DataFrame) \
            -> (pd.DataFrame, pd.DataFrame):
        try:
            preds = obj.predict_proba(x)
        except (TypeError, ValueError, DeprecationWarning) as e:
            if 'Expected 2D' in str(e) and len(x.shape) == 1:
                # print(e)
                print('=== 1D Array Error Caught. Reshaping array to overcome the error. ===')
                x = x.reshape(-1, 1)
                return self.resilient_predict_probs(obj, x)
            elif "dense" in str(e) and sparse.issparse(x):
                x = x.to_dense()
                self.use_sparse = False
                return self.resilient_predict_probs(obj, x)
            elif hasattr(obj, 'n_jobs') and obj.n_jobs != 1:
                obj.n_jobs = 1
                return self.resilient_predict_probs(obj, x)
            elif 'infs or NaNs' in str(e) and x.dtype == np.float64:
                x = x.astype(np.float32)
                return self.resilient_predict_probs(obj, x)
            elif 'infs or NaNs' in str(e) and x.dtype == np.float32:
                x = x.astype(np.float16)
                return self.resilient_predict_probs(obj, x)
            else:
                raise

        return pd.DataFrame(data=preds, index=x.index), x

    def resilient_inverse_transform(self,
                                    model: sk_prep.LabelEncoder,
                                    preds: Union[pandas_series_objs]) \
            -> pd.DataFrame:
        try:
            preds_str = model.inverse_transform(preds.astype(int))
        except TypeError as e:  # This will handle a bug with using a Numpy ndarray as an index. FUCKING HELL.
            print(e)
            preds = preds.tolist()
            preds_str = [model.classes_[i] for i in preds]
            pass

        preds_df = pd.DataFrame(data=preds_str, columns=preds.columns)
        return preds_df

    def train_predictive_model(self,
                               model: Model,
                               x_columns: list,
                               x_train: Union[pandas_dataframe_objs],
                               y_train: Union[pandas_dataframe_objs]) -> Model:
        if model.is_custom:
            if hasattr(model.model_class, 'fit'):
                clf = model.model_class
                model.model_class = type(clf)
        else:
            if model.model_class == 'rfor_c':
                clf = RandomForestClassifier(n_estimators=31)
            elif model.model_class == 'rfor_r':
                clf = RandomForestRegressor(n_estimators=31)
            elif model.model_class == 'rtree':
                clf = RandomTreesEmbedding()
            elif model.model_class == 'etree_r':
                clf = ExtraTreesRegressor()
            elif model.model_class == 'etree_c':
                clf = ExtraTreesClassifier()
            elif model.model_class == 'logit':
                clf = LogisticRegression(solver='lbfgs')
            elif model.model_class == 'linreg':
                clf = LinearRegression()
            elif model.model_class == 'ridge_r':
                clf = Ridge()
            elif model.model_class == 'ridge_c':
                clf = RidgeClassifier()
            elif model.model_class == 'lars':
                clf = Lars()
            elif model.model_class == 'gauss_proc_c':
                clf = GaussianProcessClassifier()
            elif model.model_class == 'gauss_proc_r':
                clf = GaussianProcessRegressor()
            elif model.model_class == 'lasso':
                clf = Lasso()
            elif model.model_class == 'lasso_lars':
                clf = LassoLars()
            elif model.model_class == 'lasso_mt':
                clf = MultiTaskLasso()
            elif model.model_class == 'omp':
                clf = OrthogonalMatchingPursuit()
            elif model.model_class == 'elastic_net':
                clf = ElasticNet()
            elif model.model_class == 'elastic_net_stacking':
                clf = ElasticNet(positive=True)  # Used positive=True to make this ideal for stacking
            elif 'neural_c' in model.model_class:
                clf = MLPClassifier(learning_rate='adaptive', early_stopping=True)
                if is_number(model.model_class[-1]):
                    nn_layers = int(model.model_class[-1])
                else:
                    nn_layers = 1
                x_levels = len(x_columns)
                layer_size = max(int(math.pow(x_levels, 0.5 / nn_layers)), 1)
                layer_sizes = [layer_size for l in range(nn_layers)]
                clf.hidden_layer_sizes = layer_sizes
            elif 'neural_r' in model.model_class:
                clf = MLPRegressor(learning_rate='adaptive', early_stopping=True)
                if is_number(model.model_class[-1]):
                    nn_layers = int(model.model_class[-1])
                else:
                    nn_layers = 1
                x_levels = len(x_columns)
                layer_size = max(int(math.pow(x_levels, 0.5 / nn_layers)), 1)
                layer_sizes = [layer_size for l in range(nn_layers)]
                clf.hidden_layer_sizes = layer_sizes
            elif model.model_class == 'rnn_c':
                rnn_size = 3
                # x_levels = len(x_columns)
                clf = RNNClassifier(len(x_columns))

                # clf = RNNClassifier(rnn_size=rnn_size, n_classes=15, cell_type='gru',
                #                                      input_op_fn=ft.skflow_rnn_input_fn, num_layers=1,
                #                                      bidirectional=False, sequence_length=None, steps=1000,
                #                                      optimizer='Adam', learning_rate=0.01, continue_training=True)
            elif model.model_class == 'rnn_r':
                rnn_size = 3
                # x_levels = len(x_columns)
                clf = RNNEstimator(regression_head(), len(x_columns))

                # clf = RNNEstimator(rnn_size=rnn_size, n_classes=15, cell_type='gru',
                #                                     input_op_fn=ft.skflow_rnn_input_fn, num_layers=1,
                #                                     bidirectional=False, sequence_length=None, steps=1000,
                #                                     optimizer='Adam', learning_rate=0.01, continue_training=True)
            elif model.model_class == 'svc':
                clf = SVC()
            elif model.model_class == 'svr':
                clf = SVR()
            elif model.model_class == 'nu_svc':
                clf = NuSVC()
            elif model.model_class == 'nu_svr':
                clf = NuSVR()
            elif model.model_class == 'gbc':
                clf = GradientBoostingClassifier(n_estimators=int(round(x_train.shape[0] / 20, 0)))
            elif model.model_class == 'gbr':
                clf = GradientBoostingRegressor(n_estimators=int(round(x_train.shape[0] / 20, 0)))
            elif model.model_class == 'abc':
                clf = AdaBoostClassifier(n_estimators=int(round(x_train.shape[0] / 20, 0)))
            elif model.model_class == 'knn_c':
                clf = KNeighborsClassifier()
            elif model.model_class == 'knn_r':
                clf = KNeighborsRegressor()
            elif model.model_class == 'linear_svc':
                clf = LinearSVC()
            elif model.model_class == 'sgd_c':
                clf = SGDClassifier()
            elif model.model_class == 'sgd_r':
                clf = SGDRegressor()
            elif model.model_class == 'pass_agg_c':
                clf = PassiveAggressiveClassifier()
            elif model.model_class == 'pass_agg_r':
                clf = PassiveAggressiveRegressor()
            elif model.model_class == 'bernoulli_nb':
                clf = BernoulliNB()
            elif model.model_class == 'bernoulli_rbm':
                clf = BernoulliRBM()
            elif model.model_class == 'multinomial_nb':
                clf = MultinomialNB()
            elif model.model_class == 'nearest_centroid':
                clf = NearestCentroid()
            elif model.model_class == 'xgb_r':
                clf = xgboost.XGBRegressor()
            elif model.model_class == 'xgb_c':
                clf = xgboost.XGBClassifier()
            else:
                raise ValueError(f'Incorrect model_type given. Cannot match [{model.model_class}] to a model.')

            if 'max_iter' in clf.get_params().keys():
                if isinstance(clf, (MLPClassifier, MLPRegressor, LogisticRegression)):
                    clf.max_iter = int(10e5)  # AKA 100 Thousand
                else:
                    clf.max_iter = int(10e7)  # AKA 10 Million

        if isinstance(clf, (GradientBoostingRegressor, GradientBoostingClassifier,
                            KNeighborsClassifier, KNeighborsRegressor, SVR, SVC
                            , MultiTaskLasso, LassoLars)):
            self.use_sparse = False
            if isinstance(x_train, pd.SparseDataFrame):
                x_train = x_train.to_dense()

        if 'random_state' in clf.get_params().keys():
            clf.random_state = 555
        if 'verbose' in clf.get_params().keys():
            clf.verbose = self.verbose
        if 'probability' in clf.get_params().keys():
            clf.probability = True

        # print("NaN in x_train: %s" % np.isnan(x_train.data).any())
        # print("NaN in y_train: %s" % np.isnan(y_train.data).any())

        print("\n----- Training Predictive Model [type: {0}] [usage: {1}]-----"
              .format(model.model_class, model.model_usage))

        if not model.is_custom:
            grid_param_dict = dict()

            # SOLVER
            if 'solver' in clf.get_params().keys():
                if isinstance(clf, (MLPRegressor, MLPClassifier)):
                    grid_param_dict['solver'] = ['adam',
                                                 'lbfgs']  # 'sgd' tends to crash the system when used parallel. lbfgs
                # elif isinstance(clf, (LogisticRegression)):
                #     grid_param_dict['solver'] = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

            # ACTIVATION
            if 'activation' in clf.get_params().keys():
                # Could also use 'identity' and 'tanh' (tanh tends to crash the system w/ parallel though)
                grid_param_dict['activation'] = ['logistic', 'relu']

            # CLASS_WEIGHT
            if 'class_weight' in clf.get_params().keys():
                clf.class_weight = 'balanced'

            # BOOTSTRAP
            if 'bootstrap' in clf.get_params().keys():
                clf.bootstrap = True

            # HIDDEN LAYER SIZES
            # if 'hidden_layer_sizes' in clf.get_params().keys():
            #     clf.hidden_layer_sizes = (int(len(x_columns) / 2),)

            # ALPHAS
            if 'alpha' in clf.get_params().keys():
                if isinstance(clf, GradientBoostingRegressor):
                    vals = [1e-10, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 1 - 1e-10]
                elif isinstance(clf, RidgeClassifier):
                    vals = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
                    # vals = [1]
                elif isinstance(clf, (GaussianProcessClassifier, GaussianProcessRegressor)) \
                        and 'darwin' in platform.system().lower():
                    vals = [1e-6, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-6]
                else:
                    vals = [1e-8, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-8]
                # if hasattr(clf, 'learning_rate') and clf.learning_rate == 'optimal':
                #     del vals[-1]
                grid_param_dict['alpha'] = vals

            # TOLS
            # if 'tol' in clf.get_params().keys():
            #     grid_param_dict['tol'] = [0.1, 0.01, 0.001, 0.0001, 0.00001]

            # N_NEIGHBORS
            if 'n_neighbors' in clf.get_params().keys():
                grid_param_dict['n_neighbors'] = [2, 3, 4, 5, 6, 7, 8, 9]

            # SELECTION
            if 'selection' in clf.get_params().keys():
                grid_param_dict['selection'] = ['cyclic', 'random']

            # L1_RATIO
            if 'l1_ratio' in clf.get_params().keys():
                grid_param_dict['l1_ratio'] = [.01, .1, .5, .7, .9, .95, .99, 1]

            # GAMMA
            if 'gamma' in clf.get_params().keys():
                gamma_range = np.logspace(-9, 3, 13)
                grid_param_dict['gamma'] = gamma_range

            # C
            if 'C' in clf.get_params().keys():
                C_range = np.logspace(0, 10, 11)
                grid_param_dict['C'] = C_range

            # LOSSES
            if 'loss' in clf.get_params().keys():
                if isinstance(clf, (PassiveAggressiveClassifier)):
                    grid_param_dict['loss'] = ['hinge', 'squared_hinge']
                elif isinstance(clf, (PassiveAggressiveRegressor)):
                    grid_param_dict['loss'] = ['epsilon_insensitive', 'squared_epsilon_insensitive']
                elif isinstance(clf, (SGDClassifier, SGDRegressor)):
                    grid_param_dict['loss'] = ['squared_loss', 'huber', 'epsilon_insensitive',
                                               'squared_epsilon_insensitive']
                else:
                    try:
                        grid_param_dict['loss'] = clf._SUPPORTED_LOSS
                    except:
                        pass

            # LOSSES
            if 'penalty' in clf.get_params().keys():
                if isinstance(clf, (SGDClassifier, SGDRegressor)):
                    if len(np.unique(y_train)) == 2:
                        grid_param_dict['penalty'] = ['elasticnet']
                    else:
                        grid_param_dict['penalty'] = ['l2']

            # WEIGHTS
            if 'weights' in clf.get_params().keys():
                if isinstance(clf, (KNeighborsClassifier, KNeighborsRegressor)):
                    grid_param_dict['weights'] = ['uniform', 'distance']
                else:
                    print('Unspecified parameter "weights" for ', type(clf))

            # P
            if 'p' in clf.get_params().keys():
                if isinstance(clf, (KNeighborsClassifier, KNeighborsRegressor)):
                    grid_param_dict['p'] = [1, 2, 3]
                else:
                    print('Unspecified parameter "p" for ', type(clf))

            # KERNELS
            if 'kernel' in clf.get_params().keys():
                if isinstance(clf, (GaussianProcessRegressor, GaussianProcessClassifier)):
                    grid_param_dict['kernel'] = [RBF(), RationalQuadratic(), WhiteKernel(), ConstantKernel()]
                elif isinstance(clf, (SVC, SVR)):
                    # LINEAR IS 'Work in progress.' as of 0.19
                    grid_param_dict['kernel'] = ['poly', 'rbf', 'sigmoid']  # Removed 'linear'
                else:
                    print('Unspecified parameter "kernel" for ', type(clf))

            # NU
            if 'nu' in clf.get_params().keys():
                grid_param_dict['nu'] = [1e-10, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-10]

            # METRICS
            if 'metric' in clf.get_params().keys():
                if isinstance(clf, NearestCentroid):
                    grid_param_dict['metric'] = ['euclidean', 'manhattan']
                elif isinstance(clf, (KNeighborsRegressor, KNeighborsClassifier)):
                    grid_param_dict['metric'] = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
                    # NOTE: minkowski doesn't seem to work in here. Might be a bug in sklearn 0.19.
                    # NOTE: I deliberately left out mahalanobis because it was too much trouble.
                else:
                    print('Unspecified parameter "metric" for ', type(clf))

            while True:
                try:
                    # grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, n_jobs=-1)
                    if not self.cross_val_model:
                        if hasattr(clf, 'classes_'):
                            self.cross_val_model = RepeatedStratifiedKFold(n_splits=self.cross_val_iters[0],
                                                                           n_repeats=self.cross_val_iters[1],
                                                                           random_state=555)
                        else:
                            self.cross_val_model = RepeatedKFold(n_splits=self.cross_val_iters[0],
                                                                 n_repeats=self.cross_val_iters[1],
                                                                 random_state=555)

                    if self.retrain_model and \
                            (model.trained_model is not None) and \
                            (np.testing.assert_equal(model.grid_param_dict, grid_param_dict) is None) and \
                            (model.x_columns == x_columns):

                        clf = model.trained_model
                        grid_param_dict = {}

                    elif isinstance(clf, (RNNEstimator, RNNClassifier)):

                        clf = self.resilient_fit(clf, x_train.astype(float), y_train.astype(float))
                        grid_param_dict = {}

                    else:

                        if 'windows' in platform.system().lower():  # or isinstance(clf,(MLPClassifier, MLPRegressor)):
                            grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, cv=self.cross_val_model)
                        else:
                            grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, cv=self.cross_val_model,
                                                n_jobs=-1 if self.allow_multiprocessing else 1)

                        grid = self.resilient_fit(grid, x_train.astype(float), y_train.astype(float))

                        print(grid)
                        # summarize the results of the grid search
                        print('Grid Regression Best Score:', grid.best_score_)
                        print('Grid Regression Best Estimator:', grid.best_estimator_)
                        clf = grid.best_estimator_

                        for k, v in grid.cv_results_.items():
                            if isinstance(v, (list, np.ndarray)) and all(
                                    [isinstance(v1, (float, complex, int, Number)) for v1 in v]):
                                v = mean(v)
                            print('{0}\t: {1}'.format(k, v))

                    if clf is not None:
                        model.trained_model = clf
                        model.grid_param_dict = grid_param_dict
                        model.x_columns = x_columns

                    else:
                        raise ValueError('GridSearchCV of model [{0}] did not produce a best estimator'.format(model))
                    break

                except ValueError as e:
                    if 'Invalid parameter ' in str(e):
                        grid_param_dict.pop(str(e).split(' ')[2])
                        pass
                    else:
                        raise e
        else:
            if self.retrain_model and \
                    (model.trained_model is not None) and \
                    (model.x_columns == x_columns) and \
                    (clf.get_params() == model.trained_model.get_params()):
                clf = model.trained_model
            else:
                clf = self.resilient_fit(clf, x_train, y_train)
                model.trained_model = clf
                model.x_columns = x_columns

        print("------ Model Training Complete [{0}] ------\n".format(model.model_class))

        return model

    def train_models(self,
                     df: pd.DataFrame,
                     field_names: dict
                     ):
        trained_models = OrderedDict()
        for f, t in list(field_names.items()):
            if self.verbose:
                print("VECTORIZING: ", f)

            if isinstance(t, (list, np.ndarray)):
                t = t[t != 'nan']
                vectorizer = sk_prep.LabelEncoder()
                vectorizer.fit(t)

            elif t == 'cat':
                df[f] = df[f].astype(np.str)
                mkdict = lambda row: dict((col, row[col].replace(' ', '_')) for col in [f])
                vectorizer = sk_feat.DictVectorizer(sparse=False)
                vectorizer.fit(df.apply(mkdict, axis=1))

            elif t == 'doc':
                vectorizer = sk_text.TfidfVectorizer(stop_words='english', analyzer='word', lowercase=True,
                                                     min_df=0.01)
                vectorizer.fit(filter(None, map(str.strip, df[f][~df[f].isnull()].values.astype(str))))
                # print("Feature Names for TF/IDF Vectorizer\n", *[v.encode('ascii', errors='ignore')
                #       .decode('utf-8', errors='ignore') for v in vectorizer.get_feature_names()])

            elif t in 'num':
                if self.use_sparse:
                    vectorizer = sk_prep.MaxAbsScaler()
                    # vectorizer = sk_prep.StandardScaler(with_mean=False)
                else:
                    vectorizer = sk_prep.StandardScaler()
                    # vectorizer = sk_prep.MinMaxScaler()
                vectorizer.fit(df[f].apply(pd.to_numeric, errors='coerce').fillna(0)
                               .values.astype(np.float64).reshape(-1, 1))

            elif t == 'num_noscale':
                vectorizer = None

            else:
                raise ValueError('Invalid column type provided. Choose between: \'num\', \'cat\', and \'doc\'.')
            trained_models[f] = vectorizer

        return trained_models

    def matrix_vstack(self, m: tuple, return_sparse: bool = None):
        if sum([sparse.issparse(d) for d in list(m)]) == 1:
            if self.use_sparse:
                m = [sparse.csc_matrix(d) if not sparse.issparse(d) else d for d in list(m)]
            else:
                m = [d.toarray() if sparse.issparse(d) else d for d in list(m)]
            m = tuple(m)

        if self.use_sparse:
            m = sparse.vstack(m)
            if return_sparse is False:
                m = m.toarray()
        else:
            m = np.vstack(m)
            if return_sparse:
                m = sparse.csc_matrix(m)
        return m

    def matrix_hstack(self, m: tuple, return_sparse: bool = None):
        if sum([sparse.issparse(d) for d in list(m)]) == 1:
            if self.use_sparse:
                m = [sparse.csc_matrix(d) if not sparse.issparse(d) else d for d in list(m)]
            else:
                m = [d.toarray() if sparse.issparse(d) else (d.values.reshape(-1, 1) if len(d.shape) == 1 else d) for d in
                     list(m)]
            m = tuple(m)

        if self.use_sparse:
            m = sparse.hstack(m)
            if return_sparse is False:
                m = m.toarray()
        else:
            m = np.hstack(m)
            if return_sparse:
                m = sparse.csc_matrix(m)
        return m


def fix_np_nan(m: Union[pandas_dataframe_objs + (list,)]) -> Union[pandas_dataframe_objs + (np.ndarray,)]:
    if isinstance(m, pandas_dataframe_objs):
        m.loc[:, :] = np.nan_to_num(m.values)
    elif sparse.issparse(m):
        m.data = np.nan_to_num(m.data)
    else:
        m = np.nan_to_num(m)
    return m


def is_number(s) -> bool:
    try:
        float(s)  # for int, long and float
    except ValueError:
        return False
    return True


def get_unique_name(n: str, vals: list):
    if n in vals:
        n = n + '_id1'
        while n in vals:
            n = '{0}{1}'.format(n[:-1], int(n[-1]) + 1)
    return n


def print_sp500(df: pd.DataFrame):
    if any([x == 1 for x in df.apply(pd.Series.nunique, axis=1)]):
        lol = 5
    if df['sp500_40qtr'].mean() < -10.0:
        lol = 1
    if 'sp500_40qtr_t' in df.columns:
        print(ft.str_as_header('FUCKING TESTING {0} (NULLS: {1})'.format(df['sp500_40qtr_t'].mean(), df['sp500_40qtr_t'].isnull().sum())))
    elif 'sp500_40qtr' in df.columns:
        print(ft.str_as_header('FUCKING TESTING {0} (NULLS: {1})'.format(df['sp500_40qtr'].mean(), df['sp500_40qtr'].isnull().sum())))
    else:
        raise ValueError('SP500 column not found in DF')


def update_df_outer(first: Union[pandas_dataframe_objs], other: Union[pandas_dataframe_objs]):
    for c in other.columns.tolist():
        if c not in first.columns.tolist():
            first[c] = np.NaN
    for i in first.index.values:
        if i not in other.index:
            other = other.reindex(first.index)
            break

    # print(ft.str_as_header('DATA CHECKING AT END OF "update_df_outer" FUNCTION'))
    # print(' === Other DF === \n{0}'.format(other[other.index.duplicated()]))
    # print(' === DF to update with other DF === \n{0}'.format(first[first.index.duplicated()]))
    # print('Original Size: [This DF - {0}] [Other DF - {1}]'.format(first.shape, other.shape))
    first.update(other)
    # print('New Size: [This DF - {0}]'.format(first.shape))


def get_decimals(vals: np.ndarray):
    max_decimals = max([len(v.split('.')[-1]) for v in vals if (not np.isnan(v)) and '.' in v])
    # l = [v.split('.')[-1]]
    # # x = str(x).rstrip('0')  # returns '56.001'
    # x = decimal.Decimal(x)  # returns Decimal('0.001')
    # x = x.as_tuple().exponent  # returns -3
    # x = abs(x)
    return max_decimals


def reduce_variance_corr(df: pd.DataFrame, x_column_names: list, y_column_names: list, max_corr_val: float)\
        -> list:
    print('Removing one variable for each pair of variables with correlation greater than [{0}]'.format(max_corr_val))
    # Creates Correlation Matrix and Instantiates
    corr_matrix = df.loc[:, x_column_names].astype(float).corr(method='pearson')
    # corr_matrix = df.loc[:, x_column_names].astype(float).corr()
    # for v in fields:
    # print('Field [{0}] Length: {1}'.format(v, len(df[v].unique())))
    # if len(df[v].unique()) <= 10:
    #     print(df[v].unique())
    drop_cols = set()
    if max_corr_val <= 0.:
        max_corr_val = 0.8

    # Determine the p-values of the dataset and when a field must be dropped, prefer the field with the higher p-value
    if len(np.unique(df.loc[:, y_column_names])) == 2:
        scores, p_vals = sk_feat_sel.f_classif(df.loc[:, x_column_names], df.loc[:, y_column_names])
    else:
        scores, p_vals = sk_feat_sel.f_regression(df.loc[:, x_column_names], df.loc[:, y_column_names])

    # Iterates through Correlation Matrix Table to find correlated columns
    for i, v in enumerate(x_column_names[:-1]):
        i2, c = sorted([(i2 + 1, v2) for i2, v2 in enumerate(corr_matrix.iloc[i, i + 1:])], key=lambda tup: tup[1])[-1]

        if c > max_corr_val:
            if p_vals[i] <= p_vals[i2]:
                drop_cols.add(x_column_names[i2])
            else:
                drop_cols.add(v)

    x_column_names = [v for v in x_column_names if v not in list(drop_cols)]
    print('=== Drop of Highly-Correlated Variables is Complete ===')
    print('Dropped Fields [{0}]: {1}'.format(len(drop_cols), list(drop_cols)))
    print('Remaining Fields [{0}]: {1}'.format(len(x_column_names), x_column_names))
    return x_column_names


def reduce_variance_pca(df: pd.DataFrame, column_names: list, explained_variance: float):
    print("Conducting PCA and pruning components above the desired explained variance ratio")
    max_components = len(column_names) - 1
    if explained_variance <= 0:
        explained_variance = 0.99

    pca_model = TruncatedSVD(n_components=max_components, random_state=555)
    pca_model.fit(df.loc[:, column_names])

    initial_pca_columns = pca_model.transform(df.loc[:, column_names]).T

    # print(pca_model.components_)
    print('PCA explained variance ratios.')
    print(pca_model.explained_variance_ratio_)

    final_pca_columns = []
    sum_variance = 0.
    for idx, var in enumerate(pca_model.explained_variance_ratio_):
        sum_variance += var
        pca_column_name = 'pca_{0}'.format(idx)
        df[pca_column_name] = initial_pca_columns[idx]
        final_pca_columns += [pca_column_name]
        if sum_variance > explained_variance:
            break

    print('Explained variance retained: {0:.2f}'.format(sum_variance))
    print('Number of PCA Fields: {0}'.format(len(final_pca_columns)))
    print('PCA Fields: {0}'.format(final_pca_columns))
    return df, final_pca_columns


def reduce_vars_corr(df: pd.DataFrame, field_names: list, max_num: float):
    num_vars = len(field_names)-1
    print('Current vars:  {0}'.format(num_vars))
    if not max_num or max_num < 1:
        if max_num == 0:
            max_num = 0.5
        max_num = int(df.shape[0]**max_num)

    print('Max allowed vars: {0}'.format(max_num))

    if num_vars > max_num:

        if df.isnull().any().any():
            imputed_df, field_names = ft.impute_if_any_nulls(df.loc[:, field_names].astype(float))
            for n in field_names:
                df[n] = imputed_df[n]
        # Creates Correlation Matrix
        corr_matrix = df.loc[:, field_names].corr()

        max_corr = [(fld, corr_matrix.iloc[i+1, :i].max()) for i, fld in ft.reverse_enumerate(field_names[1:])]
        max_corr.sort(key=lambda tup: tup[1])

        return_x_vals = [fld for fld, corr in max_corr[:max_num]]
        print('Number of Remaining Fields: {0}'.format(len(return_x_vals)))
        print('Remaining Fields: {0}'.format(return_x_vals))
        return df, return_x_vals

    return df, field_names


def reduce_vars_pca(df: pd.DataFrame, field_names: list, max_num: Union[int,float]=None):
    num_vars = len(field_names)-1
    print('Current vars: {0}'.format(num_vars))
    if not max_num or max_num == 0:
        max_num = int(df.shape[0]**0.5)

    print('Max allowed vars: {0}'.format(max_num))

    if num_vars > max_num:
        print("Conducting PCA and pruning components above the desired explained variance ratio")
        pca_model = TruncatedSVD(n_components=max_num, random_state=555)

        x_names_pca = []
        x_results = pca_model.fit_transform(df.loc[:, field_names]).T
        # print(pca_model.components_)
        print('PCA explained variance ratios.')
        print(pca_model.explained_variance_ratio_)

        sum_variance = 0
        for idx, var in enumerate(pca_model.explained_variance_ratio_):
            sum_variance += var
            pca_name = 'pca_{0}'.format(idx)
            df[pca_name] = x_results[idx]
            x_names_pca.append(pca_name)
            if num_vars <= max_num:
                break
        print('Explained variance retained: {0:.2f}'.format(sum_variance))
        print('Number of PCA Fields: {0}'.format(len(x_names_pca)))
        print('PCA Fields: {0}'.format(x_names_pca))
        return df, x_names_pca
    return df, field_names


def remove_high_vif(X: pd.DataFrame, max_num: Union[int,float]=None):
    num_vars = X.shape[1]
    colnames = X.columns.values
    if not max_num or max_num == 0:
        max_num = round(np.power(X.shape[0], 0.3), 0)

    if num_vars > max_num:
        print('Removing variables with high VIF. New variable qty will be: [{0}]'.format(max_num))

        # from joblib import Parallel, delayed
        while num_vars > max_num:
            # vif = Parallel(n_jobs=-1, verbose=-1)(delayed(variance_inflation_factor)(X.loc[:, colnames].values, ix) for ix in range(X.loc[:, colnames].shape[1]))
            vif = [variance_inflation_factor(X.loc[:, colnames].values, ix) for ix in range(X.loc[:, colnames].shape[1])]

            maxloc = vif.index(max(vif))
            print('dropping \'' + X.loc[:, colnames].columns[maxloc] + '\' at index: ' + str(maxloc))
            del colnames[maxloc]

        print('Remaining variables:')
        print(colnames)

    return X.loc[:, colnames], colnames.tolist()


def print_model_results(clf,
                        x_df: Union[pandas_dataframe_objs + (np.ndarray,)],
                        y_df: Union[pandas_dataframe_objs + (np.ndarray,)],
                        model_outputs_dir: str,
                        selection_limit: float):
    try:
        print_data = OrderedDict()
        print_data['Variable Name'] = x_df.columns.values.tolist()
        if hasattr(clf, 'coef_'):
            coef_fld_name = 'Coef.'
            if isinstance(clf.coef_[0], (tuple, list, np.ndarray)):
                if clf.coef_.shape[0] == 1:
                    coefs = fix_np_nan(clf.coef_[0])
                else:
                    coefs = fix_np_nan([v[0] for v in clf.coef_])
            else:
                coefs = fix_np_nan([v for v in clf.coef_])
            if isinstance(coefs, (np.ndarray)):
                coefs = coefs.tolist()
            elif sparse.issparse(coefs):
                coefs = coefs.tolist()[-1].data

            coefs = [f'{v:.2E}' for v in coefs]
            sigmoid_coef = [f'{ft.sigmoid(float(v)):.3f}' for v in coefs]
            try:
                exp_coef = [f'{exp(float(v)):.3f}' for v in coefs]
            except OverflowError:
                print('Coefficients too large to take the natural log. Skipping.')
                exp_coef = [f'N/A' for v in coefs]
                pass

        elif hasattr(clf, 'feature_importances_'):
            coef_fld_name = 'Feat Imp.'
            coefs = [f'{v:.1%}' for v in fix_np_nan(clf.feature_importances_).tolist()]
            sigmoid_coef = [f'N/A' for v in coefs]
            exp_coef = [f'N/A' for v in coefs]
        else:
            raise ValueError('No coef_ or feature_importances_ attribute in this model. Skipping.')

        print("Total Coefficients/Features:  ", len(coefs))
        print_data[coef_fld_name] = coefs
        print_data[coef_fld_name + ' (Sigmoid)'] = sigmoid_coef
        print_data[coef_fld_name + ' (Exp)'] = exp_coef

        if len(np.unique(y_df)) == 2:
            scores, p_vals = sk_feat_sel.f_classif(x_df, y_df)
        else:
            scores, p_vals = sk_feat_sel.f_regression(x_df, y_df, center=False)

        print_data['Score'] = [f'{float(v):.3f}' for v in scores.tolist()]
        print_data['P-Value'] = [f'{float(v):.4f}' for v in p_vals.tolist()]

        sig_lvls = list()
        for v in p_vals:
            signif_str = ''
            for lvl in (0.1, 0.05, 0.01, 0.001):
                if float(v) <= lvl:
                    signif_str += '*'
            sig_lvls += [signif_str]

        print_data['Signif. Lvl'] = sig_lvls

        if len(set(map(len, print_data.values()))) == 1:
            all_coefs = pd.DataFrame.from_dict(print_data)
        else:
            raise ValueError('Cannot print dataframe with coefficient information.\n'
                             'All the coefficient value series are not the same length!!!!')

        all_coefs.sort_values('P-Value', ascending=True, inplace=True)

        print(f'Outputting Model Coefficient Information to {model_outputs_dir}')
        all_coefs_file_name = 'all_coefs'
        while True:
            try:
                all_coefs.to_csv(join(model_outputs_dir, all_coefs_file_name + '.csv'), index=True,
                                 index_label=['index1'], header=True, encoding='utf_8')
                del all_coefs_file_name
                break
            except PermissionError as e:
                print(e)
                all_coefs_file_name += f' ({dt.now().strftime("%Y-%m-%d %H:%M:%S")})'
                pass

        nonzero_coef_mask = all_coefs[coef_fld_name].str.replace('%', '').astype(float) != 0
        signif_var_mask = all_coefs['P-Value'].astype(float) <= selection_limit
        print_coefs = all_coefs.loc[nonzero_coef_mask & signif_var_mask, :]

        print_table = PrettyTable(print_coefs.columns.tolist())
        for _, r in print_coefs.iterrows():
            print_table.add_row(r)

        print(print_table.get_string())

        try:
            with open(join(model_outputs_dir, 'coef_info.txt'), mode='wt', encoding='utf_8') as f:
                f.writelines(print_table.get_string())
        except UnicodeDecodeError as e:
            print(e)
            pass

    except ValueError as e:
        print(e)
        pass
