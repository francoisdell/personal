__author__ = 'Andrew Mackenzie'
import os
import sys

if os.path.abspath(os.pardir) not in sys.path:
    sys.path.append(os.path.abspath(os.pardir))

# from PyInstaller import compat
# mkldir = os.path.join(compat.base_prefix, "Library", "bin")
# binaries = [(os.path.join(mkldir, mkl), '') for mkl in os.listdir(mkldir) if mkl.startswith('mkl_')]
# if not mkldir in os.environ['PATH']:
#         os.environ['PATH'] = mkldir + os.pathsep + os.environ['PATH']
# for b in binaries:
#     if not b[0] in os.environ['PATH']:
#         os.environ['PATH'] = b[0] + os.pathsep + os.environ['PATH']

from typing import Union
from math import exp
import math
from Settings import Settings
from collections import OrderedDict
import numpy as np
import prettytable
import platform
import pandas as pd
import pickle
import bisect
import decimal
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
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomTreesEmbedding
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC, SVR, NuSVR, NuSVC
from sklearn.metrics import r2_score
from scipy.special import expit
from numbers import Number
from sklearn import metrics
import warnings
from statistics import mean
import shutil


pd.options.mode.chained_assignment = None
pd.set_option('display.height', 1000)
pd.set_option('display.width', shutil.get_terminal_size()[0])
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 0)
pd.set_option('display.max_colwidth', 100)


class ModelSet:
    def __init__(self,
                 final_models: Union[str, list],
                 initial_models: Union[str, list]=list(),
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
                 trained_model: object=None,
                 grid_param_dict: dict=None,
                 x_columns: list=None):

        self.model_class = model_class
        self.model_usage = model_usage
        self.trained_model = trained_model
        self.grid_param_dict = grid_param_dict
        self.is_custom = not isinstance(model_class, str)

    def get_info(self):
        return self.model_class, self.model_usage

    def __str__(self):
        return self.model_class + '_' + self.model_usage


class Model_Builder:
    def __init__(self
            , df: pd.DataFrame
            , x_fields: dict
            , y_field: Union[dict, OrderedDict]
            , model_type: Union[str, list, ModelSet]
            , report_name: str
            , show_model_tests: bool=False
            , retrain_model: bool=False
            , selection_limit: float=5.0e-2
            , predict_all: bool=False
            , verbose: bool=True
            , train_pct: float=0.7
            , random_train_test: bool=True
            , max_train_data_points: int=50000
            , pca_explained_var: float=1.0
            , stack_include_preds: bool=True
            , final_include_data: bool=True
            , cross_val_iters: tuple=(3, 2)
            , cross_val_model=None
            , use_test_set: bool=False
            , auto_reg_periods: int=0):
        
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
        self.use_sparse = True
        self.s = Settings(report_name=report_name)
        self.use_test_set = use_test_set
        self.auto_reg_periods = auto_reg_periods

    def predict(self) -> pd.DataFrame:
    
        s = Settings(report_name=self.report_name)
        final_file_dir = s.get_default_data_dir()
        # os.makedirs(final_file_dir, exist_ok=True)
    
        if not self.verbose:
            # warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=exceptions.ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
    
        elif isinstance(self.model_type, list):
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

            self.train_pct = min(self.train_pct, self.max_train_data_points / (len(self.df[key]) - mask_train_test_qty_true))
            test_pct = min((1-self.train_pct), self.max_train_data_points / (len(self.df[key]) - mask_train_test_qty_true))
    
            if self.verbose:
                print("Train data fraction:", self.train_pct)
                print("Test data fraction:", test_pct)
    
            if self.random_train_test:
                mask = np.random.rand(len(self.df[key]))
                mask_train = [a and b for a, b in zip(mask <= self.train_pct, mask_train_test != False)]
                mask_test = [a and b for a, b in zip(mask > (1-test_pct), mask_train_test != False)]
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
    
            if not self.use_test_set:
                print("Don't need to split the set into train/validate since we're using cross validation.")
                mask_train = mask_train_test
    
            # print("Mask_Train Value Counts\n", pd.Series(mask_train).value_counts())
            # print("Mask_Test Value Counts\n", pd.Series(mask_test).value_counts())
        # x_mappings, x_columns, x_vectors = set_vectors(df.loc[mask_train_test,], x_fields)
        # y_mappings, y_vectors = set_vectors(df, y_field, is_y=True)
    
        try:
            if self.retrain_model:
                raise ValueError('"retrain_model" variable is set to True. Training new preprocessing & predictive models.')
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
            retrain_model = True

        # If autoregression has been requested, add the autoregression varibales
        # if self.auto_reg_periods > 0:
        #     for y_field, y_type in self.y_field.items():
        #         for lag in range(self.auto_reg_periods):
        #             y_field_lag_name = y_field + '_lag{0}'.format(lag+1)
        #             self.df[y_field_lag_name] = self.df[y_field].shift(lag-1)
        #             if np.isnan(self.df[y_field_lag_name].iloc[0]):
        #                 self.df[y_field_lag_name].iloc[0] = self.df[y_field_lag_name].mean()
        #             self.x_fields[y_field_lag_name] = y_type

        # CHANGE THE 'cat' to a list of all possible values in the y field. This is necessary because the LabelEncoder
        # that will encode the Y variable values can't handle never-before-seen values. So we need to pass it every
        # possible value in the Y variable, regardless of whether it appears in the train or test subsets.
        for k, v in self.y_field.items():
            if v == 'cat':
                self.y_field[k] = self.df[k].unique().astype(str)
    
        y_mappings = self.train_models(self.df[mask_train], self.y_field)
        _, y_train, y_mappings = self.get_vectors(self.df[mask_train], self.y_field, y_mappings, is_y=True)

        x_train, self.x_fields, x_columns, x_mappings = \
            self.get_fields(self.df,
                            self.x_fields,
                            y_train,
                            mask_train,
                            self.selection_limit)
    
        pred_x_fields = dict()
        pred_x_mappings = dict()
        if self.verbose:
            print('FIELDS\n', np.asarray(list(self.x_fields.keys())))

        for idx, model in enumerate(self.model_type.initial_models):



            model, x_train = self.train_predictive_model(model,
                                                        self.retrain_model,
                                                        self.cross_val_iters,
                                                        self.cross_val_model,
                                                        x_columns,
                                                        x_train,
                                                        y_train)

            clf = model.trained_model

            max_slice_size = 100000
            y_all = list()
            mask_all = np.asarray([True for v in range(self.df.shape[0])], dtype=np.bool)
            # y_all_probs = list()
            for s in range(0, int(math.ceil(len(self.df.index) / max_slice_size))):
                min_idx = s * max_slice_size
                max_idx = min(len(self.df.index), (s + 1) * max_slice_size)
                print("Prediction Iteration #%s: min/max = %s/%s" % (s, min_idx, max_idx))

                df_valid_iter = self.df[mask_all].iloc[min_idx:max_idx]
                x_valid_columns, x_all, x_mappings = self.get_vectors(df_valid_iter, self.x_fields, x_mappings)
                x_all = fix_np_nan(x_all)

                # print(x_validate)
                calculate_probs = hasattr(clf, 'classes_') \
                                  and hasattr(clf, 'predict_proba') \
                                  and not (hasattr(clf, 'loss') and clf.loss == 'hinge')

                preds, x_all = self.resilient_predict(clf, x_all)
                if calculate_probs:
                    pred_probs, x_all = self.resilient_predict_probs(clf, x_all)
                y_all.append(preds)

            # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
            y_all = np.hstack(y_all).reshape(-1, 1)

            # Generate predictions for this predictive model, for all values. Add them to the DF so they can be
            #  used as predictor variables (e.g. "stacked" upon) later when the next predictive model is run.
            for k, v in self.y_field.items():
                # y_all = np.atleast_1d(y_all)
                preds = y_mappings[k].inverse_transform(y_all)
                print('INTERMEDIATE PREDICTIONS')
                print("First 10: ", *preds[:10])
                print("Last 10: ", *preds[-10:])

                new_x_fields = {}
                if calculate_probs:
                    prob_field_names = [k + '_' + model.model_class + '_prob_' + f.lower().replace(' ', '_') for f in
                                        list(y_mappings[k].classes_)[:pred_probs.shape[1]]]
                    prob_field_names = [get_unique_name(v, self.df.columns.values) for v in prob_field_names]
                    df_pred = pd.DataFrame(data=pred_probs
                                           , columns=prob_field_names
                                           , index=df_valid_iter.index)
                    df_pred = df_pred.loc[:, df_pred.columns.values[:-1]]
                    for f in df_pred.columns.values:
                        new_x_fields[f] = 'num'
                    # self.df = self.df.join(df_preds, how='inner')
                    print('Unique probabilities: ', len(np.unique(pred_probs)))
                    print('Probability variance: ', np.var(pred_probs))

                else:
                    pred_field_name = get_unique_name('{0}_pred_{1}'.format(k, model.model_class), self.df.columns.values)
                    df_pred = pd.DataFrame(data=preds
                                           , columns=[pred_field_name]
                                           , index=df_valid_iter.index)
                    # df[pred_field_name] = preds

                    new_x_fields[pred_field_name] = 'cat' if not isinstance(v, str) else v
                    print('Unique predictions: ', len(np.unique(preds)))

                self.df = self.df.join(df_pred, how='inner')
                new_x_mappings = self.train_models(self.df[mask_train], new_x_fields)

                if self.stack_include_preds:
                    self.x_fields = {**self.x_fields, **new_x_fields}
                    # x_mappings = {**x_mappings, **new_x_mappings}
                    # new_x_columns, new_x_train, x_mappings = self.get_vectors(self.df[mask_train], new_x_fields, x_mappings)
                    # x_columns += new_x_columns
                    # new_x_train = fix_np_nan(new_x_train)
                    # new_x_train = sparse.csc_matrix(np.hstack(new_x_train))
                    # x_train = self.matrix_hstack((x_train, new_x_train), return_sparse=True)
                    x_train, self.x_fields, x_columns, x_mappings = \
                        self.get_fields(self.df,
                                        self.x_fields,
                                        y_train,
                                        mask_train,
                                        self.selection_limit)
                else:
                    pred_x_fields = {**pred_x_fields, **new_x_fields}
                    pred_x_mappings = {**pred_x_mappings, **new_x_mappings}

            if hasattr(clf, 'intercept_') and (isinstance(clf.intercept_, (list)) and len(clf.intercept_) == 1):
                print('--- Model over-normalization testing ---\n'
                      'Intercept/Expit/Exp = {0} / {1} / {2}'
                      .format(format(clf.intercept_[0], '.4f')
                              , format(expit(clf.intercept_[0]), '.4f')
                              , format(exp(clf.intercept_[0])), '.4f'))


        for idx, model in enumerate(self.model_type.final_models):

            if self.final_include_data:
                self.x_fields = {**self.x_fields, **pred_x_fields}
            else:
                self.x_fields = pred_x_fields

            x_train, self.x_fields, x_columns, x_mappings = \
                self.get_fields(self.df,
                                self.x_fields,
                                y_train,
                                mask_train,
                                self.selection_limit)

            model, x_train = self.train_predictive_model(model,
                                                        self.retrain_model,
                                                        self.cross_val_iters,
                                                        self.cross_val_model,
                                                        x_columns,
                                                        x_train,
                                                        y_train)

            clf = model.trained_model
            # If PCA has been specified, convert x_fields to PCA
            if self.pca_explained_var < 1:
                self.df, x_columns = self.convert_to_pca(pca_df=self.df,
                                                    field_names=list(self.x_fields.keys()),
                                                    explained_variance=self.pca_explained_var)

            # INITIAL AND FINAL MODELS: SHOW MODEL TESTS
            if self.show_model_tests:
                print("--------------------------\n"
                      "-- TEST SET APPLICATION --\n"
                      "--------------------------")
                x_columns, x_test, x_mappings = self.get_vectors(self.df[mask_test], self.x_fields, x_mappings)
                _, y_test, y_mappings = self.get_vectors(self.df[mask_test], self.y_field, y_mappings, is_y=True)
                x_test = fix_np_nan(x_test)

                print("Total Named Columns: ", len(x_columns))

                coef_list = [['name'] + [new_colname.strip() for new_colname, show, orig_colname in x_columns]]

                if hasattr(clf, 'feature_importances_'):
                    feat_importances = ['coef'] + fix_np_nan(clf.feature_importances_).tolist()
                    print('Feature Importances:\n%s' % clf.feature_importances_, sep='\n')
                    coef_list.append(feat_importances)

                try:
                    if hasattr(clf, 'coef_'):
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
                        print("Total Coefficients/Features:  ", len(coefs))
                        # elif hasattr(clf, 'coefs_'):
                        # coefs = np.nan_to_num(np.transpose(clf.coefs_))
                        # multiplied_coefs = coefs[0][0] * coefs[1] / float(len(coefs[0]))
                        # print(multiplied_coefs.sum())
                        # print(coefs[0][0])
                        # print(coefs[1])
                        # print(len(coefs[0][0]))
                        # print(len(coefs[1]))
                        # if isinstance(coefs.tolist()[0], (list, tuple)):
                        #     coefs = [(l * [c.sum() if isinstance(c, np.ndarray) else c for c in coefs[1]]).sum() / float(len(l)) for l in coefs[0]]
                        # print('Coefficients: ', coefs)
                        # print('Coefficient Matrix Shape: ', coefs.shape)
                    else:
                        raise ValueError('No coef_ or coefs_ attrubute in this model. Skipping.')

                    if len(np.unique(y_test)) == 2:
                        scores, p_vals = sk_feat_sel.f_classif(x_test, y_test)
                    else:
                        scores, p_vals = sk_feat_sel.f_regression(x_test, y_test, center=False)

                    nonzero_coefs_mask = [v != 0 for v in coefs]
                    significant_coefs_mask = [v <= self.selection_limit for v in p_vals]

                    print_df = pd.DataFrame(list(zip([v[0] for v in x_columns]
                                                     , coefs
                                                     , scores.tolist()
                                                     , p_vals.tolist()))
                                            , columns=['Name', 'Coef', 'Score', 'P-Value'])

                    print_df = print_df.loc[
                               [v1 and v2 for v1, v2 in zip(nonzero_coefs_mask, significant_coefs_mask)], :]
                    print_df.sort_values('Coef', ascending=True, inplace=True)

                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(print_df.to_string())
                        # if nonzero_coefs_mask.value_counts().loc[True] <= 40:
                        # nonzero_coefs_mask = nonzero_coefs_mask.values

                        # print('==== Final Variables ====\n', np.asarray([v1 for v1, v2 in np.asarray(x_columns)[nonzero_coefs_mask]], dtype=np.str).sort())
                        #
                        # print("==== Coefficients ====\n", np.asarray(np.asarray(coefs)[nonzero_coefs_mask]))
                        #
                        # print('==== Scores ====\n', scores[nonzero_coefs_mask])
                        # print('==== P-Values ====\n', p_vals[nonzero_coefs_mask])

                    if len(coefs) > 0:
                        try:
                            coefs = ['coefs'] + [v for v in coefs]
                            coef_list.append(coefs)

                            expit_coefs = ['coef_expit'] + [expit(float(v)) for v in coefs[1:]]
                            coef_list.append(expit_coefs)

                            exp_coefs = ['coef_exp'] + ['%f' % exp(float(v)) for v in coefs[1:]]
                            coef_list.append(exp_coefs)
                        except OverflowError:
                            pass
                except ValueError as e:
                    print(e)
                    pass

                if len(np.unique(y_test)) == 2:
                    scores, p_vals = sk_feat_sel.f_classif(x_test, y_test)
                else:
                    scores, p_vals = sk_feat_sel.f_regression(x_test, y_test, center=False)

                scores = ['scores'] + scores.tolist()
                p_vals = ['p_values'] + p_vals.tolist()
                coef_list.append(scores)
                coef_list.append(p_vals)

                coef_list = list(map(list, zip(*coef_list)))
                for i, r in enumerate(coef_list[1:]):
                    coef_list[i + 1] = [r[0]] + [round(float(v), 4) for v in r[1:]]

                value_mask_95 = [bool(x_columns[i][1]) and (float(v) <= 0.05) for i, v in enumerate(p_vals[1:])]

                # print('PREDICTOR VARIABLES')
                # print(coef_list)

                pp2 = prettytable.PrettyTable(coef_list[0])
                for i, r in enumerate(sorted(coef_list[1:], key=lambda x: x[-1], reverse=False)):
                    if value_mask_95[i]:
                        pp2.add_row(r)

                # print('PREDICTOR VARIABLES SIGNIFICANT AT 95%')
                # print(pp2.get_string())

                # TRY TO SAVE THE COEFFICIENT LIST TO A FILE
                try:
                    all_coefs = pd.DataFrame(coef_list[1:], columns=coef_list[0])
                    all_coefs.to_csv('%s/%s' % (final_file_dir, 'all_coefs.csv'), index=True,
                                     index_label=['index1'], header=True, encoding='utf_8')
                except PermissionError as e:
                    print(e)
                    pass

                try:
                    with open('%s/%s' % (final_file_dir, 'coef_info.txt'), mode='wt', encoding='utf_8') as f:
                        f.writelines(pp2.get_string())
                except UnicodeDecodeError as e:
                    print(e)
                    pass

                if hasattr(clf, 'intercept_'):
                    intercept = clf.intercept_
                    if isinstance(clf.intercept_, (np.float64, float)):
                        intercept = [intercept]

                    print('INTERCEPT')
                    print(*intercept, sep='\n')

                    print('INTERCEPT (EXPIT)')
                    print(*[expit(v) for v in intercept], sep='\n')

                    print('INTERCEPT (EXP)')
                    print(*[exp(v) for v in intercept], sep='\n')

                preds, x_test = self.resilient_predict(clf, x_test)
                print('R2 SCORE\n', r2_score(y_test, preds))

                if hasattr(clf, 'classes_'):
                    print('CLASSES')
                    print(clf.classes_)
                    print('ACCURACY:\n', metrics.accuracy_score(y_test, preds))
                    try:
                        y_test_probs, x_test = self.resilient_predict_probs(clf, x_test)
                        if y_test_probs.shape[1] <= 2:
                            print('ROC-CURVE AOC', metrics.roc_auc_score(y_test, y_test_probs[:, 1]))

                        print("PREDICTION PROBABILITIES")
                        print(y_test_probs)
                    except (ValueError, AttributeError) as e:
                        print(e)
                        pass

                    # if not isinstance(preds, list):
                    #     preds = [preds]
                    # lol = list(enumerate(preds))
                    # maps = list(y_mappings.values())

                    # preds = [list(y_mappings.values())[index][values] for index, values in enumerate(preds)]

                    for k, v in self.y_field.items():
                        preds_str = self.resilient_inverse_transform(y_mappings[k], preds)
                        preds_names = list(set(preds_str))

                        print("Unique integers in y_test:", list(set(y_test)))
                        y_test_str = self.resilient_inverse_transform(y_mappings[k], y_test)
                        y_test_names = list(set(y_test_str))
                        # Create confusion matrix
                        conf_matrix = pd.crosstab(y_test_str, preds_str, rownames=['actual'],
                                                  colnames=['predicted'])
                        print("CONFIDENCE MATRIX")
                        print(conf_matrix)

                pass

            print("--------------------------------\n"
                  "-- VALIDATION SET APPLICATION --\n"
                  "--------------------------------")
            # CREATE X_VALIDATE
            df_validate = self.df.loc[mask_validate, :]

            # ITERATE OVER THE df_validate in groups of 100K rows (to avoid memory errors) and predict outcomes
            max_slice_size = 100000
            y_validate = list()
            y_validate_probs = list()
            for s in range(0, int(math.ceil(len(df_validate.index) / max_slice_size))):
                min_idx = s * max_slice_size
                max_idx = min(len(df_validate.index), (s + 1) * max_slice_size)
                print("Prediction Iteration #%s: min/max = %s/%s" % (s, min_idx, max_idx))

                df_valid_iter = self.df[mask_validate].iloc[min_idx:max_idx]
                x_valid_columns, x_validate, x_mappings = self.get_vectors(df_valid_iter, self.x_fields, x_mappings)

                x_validate = fix_np_nan(x_validate)

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
                preds, x_validate = self.resilient_predict(clf, x_validate)

                calculate_probs = hasattr(clf, 'classes_') \
                                  and hasattr(clf, 'predict_proba') \
                                  and not (hasattr(clf, 'loss') and clf.loss == 'hinge')
                if calculate_probs:
                    pred_probs, x_validate = self.resilient_predict_probs(clf, x_validate)
                    y_validate_probs.append(pred_probs)

                y_validate.append(preds)

            # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
            y_validate = np.hstack(y_validate).reshape(-1, 1)

            self.new_fields = list()
            if calculate_probs:
                y_validate_probs = np.vstack(y_validate_probs)
                print('ORIGINAL PROBABILITIES')
                print(*y_validate_probs[:10].round(4), sep='\n')
                print('ORIGINAL PROBABILITIES (NORMALIZED)')
                print(*[np.around(np.expm1(x), 4) for x in y_validate_probs][:10], sep='\n')

                for y_name, mapping in y_mappings.items():
                    df_probs = pd.DataFrame(data=y_validate_probs.round(4)
                                            , columns=[y_name + '_prob_' + f.lower().replace(' ', '_') for f in
                                                       list(mapping.classes_)[:y_validate_probs.shape[1]]]
                                            , index=df_validate.index)
                    # df_validate.reset_index(inplace=True, drop=True)

                    for name in df_probs.columns.values:
                        if name in df_validate.columns.values:
                            df_probs.rename(columns={name: name + '_final'}, inplace=True)
                    df_validate = df_validate.join(df_probs, how='inner')
                    # for col in df_probs.columns.values:
                    #     df_validate[col] = df_probs[col]
                self.new_fields.extend(df_probs.columns.values.tolist())

            for k, v in self.y_field.items():
                final_preds = y_mappings[k].inverse_transform(y_validate)
                print('FINAL PREDICTIONS')
                print(*final_preds[:10])
                y_pred_name = 'pred_' + k
                df_validate.loc[:, y_pred_name] = final_preds
                self.new_fields.extend([y_pred_name])

            if self.predict_all:
                return_df = df_validate
            else:
                return_df = self.df.loc[mask_validate == 0, :].append(df_validate, ignore_index=True)

            yield return_df, model.model_class


        # INITIAL AND FINAL MODELS: TRAIN ANY UNTRAINED MODELS
        # print(x_train.shape)
        if self.verbose:
            print('FIELDS\n', np.asarray(list(self.x_fields.keys())))
        # y_train, uniques_index = pd.factorize(train[y_fields])


        with open(predictive_model_file, 'wb') as pickle_file:
            pickle.dump((self.model_type,), pickle_file)
        with open(mappings_file, 'wb') as pickle_file:
            pickle.dump((x_mappings, y_mappings), pickle_file)

    def get_fields(self,
                   df,
                   fields,
                   y,
                   mask,
                   selection_limit):

        mappings = self.train_models(df[mask], fields)
        columns, x, mappings = self.get_vectors(df[mask], fields, mappings)
        x = fix_np_nan(x)

        if selection_limit < 1.0:
            print('Pruning x_fields for any variables with a p-value > {0}'.format(selection_limit))
            if len(np.unique(y)) == 2:
                scores, p_vals = sk_feat_sel.f_classif(x, y)
            else:
                scores, p_vals = sk_feat_sel.f_regression(x, y, center=False)
            for field_name in list(fields.keys()):
                xcol_indices = [idx for idx, vals in enumerate(columns) if vals[2] == field_name]
                if all(p_vals[idx] > selection_limit or p_vals[idx] == np.nan for idx in xcol_indices):
                    fields.pop(field_name)
            columns, x, mappings = self.get_vectors(df[mask], fields, mappings)
            x = fix_np_nan(x)

        return x, fields, columns, mappings

    
    def convert_to_pca(self, 
                       pca_df: pd.DataFrame,
                       field_names: list,
                       explained_variance: float
                       ) -> (pd.DataFrame, list):
        print("Conducting PCA and pruning components above the desired explained variance ratio")
        max_components = len(field_names) - 1
        pca_model = TruncatedSVD(n_components=max_components, random_state=555)
    
        x_results = pca_model.fit_transform(pca_df.loc[:, field_names]).T
        print(pca_model.components_)
        print(pca_model.explained_variance_ratio_)
    
        x_names_pca = []
        sum_variance = 0
        for idx, var in enumerate(pca_model.explained_variance_ratio_):
            sum_variance += var
            pca_name = 'pca_{0}'.format(idx)
            pca_df[pca_name] = x_results[idx]
            x_names_pca.append(pca_name)
            if sum_variance > explained_variance:
                break
        return pca_df, x_names_pca
    
    
    def resilient_fit(self, obj, x, y) -> (object, object):
        try:
            obj.fit(x, y)
        except (TypeError, ValueError) as e:
            if "dense" in str(e):
                x = x.toarray()
                self.use_sparse = False
                obj.fit(x, y)
                pass
            else:
                raise
        except DeprecationWarning as e:
            print("YOU NEED TO FIX THE BELOW ERROR SINCE IT WILL BE DEPRECATED")
            print(e)
            x_train = x.reshape(-1, 1)
            obj.fit(x_train, y)
            pass
    
        return obj, x
    
    
    def resilient_predict(self, obj, x) -> (object, object):
        try:
            preds = obj.predict(x)
        except (TypeError, ValueError) as e:
            if "dense" in str(e):
                x = x.toarray()
                self.use_sparse = False
                preds = obj.predict(x)
                pass
            else:
                raise
        except DeprecationWarning as e:
            print("YOU NEED TO FIX THE BELOW ERROR SINCE IT WILL BE DEPRECATED")
            print(e)
            x = x.reshape(-1, 1)
            preds = obj.predict(x)
            pass
    
        return preds, x
    
    
    def resilient_predict_probs(self, obj, x) -> (object, object):
        try:
            preds = obj.predict_proba(x)
        except (TypeError, ValueError) as e:
            if "dense" in str(e):
                x = x.toarray()
                self.use_sparse = False
                preds = obj.predict_proba(x)
                pass
            else:
                raise
        except DeprecationWarning as e:
            print("YOU NEED TO FIX THE BELOW ERROR SINCE IT WILL BE DEPRECATED")
            print(e)
            x = x.reshape(-1, 1)
            preds = obj.predict_proba(x)
            pass
    
        return preds, x
    
    
    def resilient_inverse_transform(self, model: sk_prep.LabelEncoder, preds: np.ndarray):
        try:
            preds_str = model.inverse_transform(preds)
        except TypeError as e:  # This will handle a bug with using a Numpy ndarray as an index. FUCKING HELL.
            print(e)
            preds = preds.tolist()
            preds_str = [model.classes_[i] for i in preds]
            pass
        return preds_str


    def train_predictive_model(self,
                               model,
                               retrain_model,
                               cross_val_iters,
                               cross_val_model,
                               x_columns,
                               x_train,
                               y_train):
        if model.is_custom:
            if hasattr(model.model_class, 'fit'):
                clf = model.model_class
                model.model_class = type(clf)
        else:
            if model.model_class == 'rfor':
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
                clf = LogisticRegression()
            elif model.model_class == 'linreg':
                clf = LinearRegression()
            elif model.model_class == 'ridge':
                clf = Ridge()
            elif model.model_class == 'ridge_c':
                clf = RidgeClassifier()
            elif model.model_class == 'lars':
                clf = Lars()
            elif model.model_class == 'gauss_proc_c':
                clf = GaussianProcessClassifier(RBF(1.0))
            elif model.model_class == 'gauss_proc_r':
                clf = GaussianProcessRegressor(RBF(1.0))
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
            elif model.model_class == 'neural_c':
                clf = MLPClassifier(learning_rate='adaptive', learning_rate_init=0.1)
            elif model.model_class == 'neural_r':
                clf = MLPRegressor(learning_rate='adaptive', learning_rate_init=0.1)
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
            else:
                raise ValueError('Incorrect model_type given. Cannot match [%s] to a model.' % model.model_class)

            if 'max_iter' in clf.get_params().keys():
                if (not clf.max_iter) or (clf.max_iter < 10000):
                    clf.max_iter = 10000

        if isinstance(clf, (GradientBoostingRegressor, GradientBoostingClassifier,
                            KNeighborsClassifier, KNeighborsRegressor, SVR, SVC
                            , MultiTaskLasso, LassoLars)):
            self.use_sparse = False
            if not isinstance(x_train, (np.ndarray)):
                x_train = x_train.toarray()
            if not isinstance(y_train, (np.ndarray)):
                y_train = y_train.toarray().reshape(-1, 1)
        else:
            self.use_sparse = True

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
                    grid_param_dict['solver'] = ['adam']  # 'sgd' tends to crash the system when used parallel. lbfgs
                    # elif isinstance(clf, (LogisticRegression)):
                    #     grid_param_dict['solver'] = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

            # ACTIVATION
            if 'activation' in clf.get_params().keys():
                # recommend against using 'logistic' for the activation. Just use a logistic regression instead.
                grid_param_dict['activation'] = ['identity', 'relu']  # tanh tends to crash the system w/ parallel

            # CLASS_WEIGHT
            if 'class_weight' in clf.get_params().keys():
                clf.class_weight = 'balanced'

            # BOOTSTRAP
            if 'bootstrap' in clf.get_params().keys():
                clf.bootstrap = True

            # ALPHAS
            if 'alpha' in clf.get_params().keys():
                if isinstance(clf, (GradientBoostingRegressor)):
                    vals = [1e-10, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 1 - 1e-10]
                elif isinstance(clf, (RidgeClassifier)):
                    # vals = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
                    vals = [1]
                else:
                    vals = [1e-10, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-10]
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
                    grid_param_dict['loss'] = ['squared_loss', 'huber', 'epsilon_insensitive'
                        , 'squared_epsilon_insensitive']
                else:
                    grid_param_dict['loss'] = clf._SUPPORTED_LOSS

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
                    grid_param_dict['kernel'] = [RBF(), RationalQuadratic(), WhiteKernel()
                        , ConstantKernel()]
                elif isinstance(clf, (SVC, SVR)):
                    grid_param_dict['kernel'] = ['linear', 'poly', 'rbf',
                                                 'sigmoid']  # LINEAR IS 'Work in progress.' as of 0.19
                else:
                    print('Unspecified parameter "kernel" for ', type(clf))

            # NU
            if 'nu' in clf.get_params().keys():
                grid_param_dict['nu'] = [1e-10, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-10]

            # METRICS
            if 'metric' in clf.get_params().keys():
                if isinstance(clf, (NearestCentroid)):
                    grid_param_dict['metric'] = ['euclidean', 'manhattan']
                elif isinstance(clf, (KNeighborsRegressor, KNeighborsClassifier)):
                    grid_param_dict['metric'] = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
                    # NOTE: wminkowski doesn't seem to work in here. Might be a bug in sklearn 0.19.
                    # NOTE: I deliberately left out seuclidean and mahalanobis because they were too much trouble.
                else:
                    print('Unspecified parameter "metric" for ', type(clf))

            while True:
                try:
                    # grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, n_jobs=-1)
                    if not cross_val_model:
                        if hasattr(clf, 'classes_'):
                            cross_val_model = RepeatedStratifiedKFold(n_splits=cross_val_iters[0], 
                                                                      n_repeats=cross_val_iters[1],
                                                                      random_state=555)
                        else:
                            cross_val_model = RepeatedKFold(n_splits=cross_val_iters[0], 
                                                            n_repeats=cross_val_iters[1],
                                                            random_state=555)

                    if 'windows' in platform.system().lower():  # or isinstance(clf,(MLPClassifier, MLPRegressor)):
                        grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, cv=cross_val_model)
                    else:
                        grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, cv=cross_val_model, n_jobs=-1)

                    if retrain_model and \
                            (model.trained_model is not None) and \
                            (np.testing.assert_equal(model.grid_param_dict, grid_param_dict) is None) and \
                            (model.x_columns == x_columns):
                        clf = model.trained_model
                    else:

                        grid, x_train = self.resilient_fit(grid, x_train, y_train)

                        print(grid)
                        # summarize the results of the grid search
                        print('Grid Regression Best Score:', grid.best_score_)
                        print('Grid Regression Best Estimator:', grid.best_estimator_)
                        clf = grid.best_estimator_
                        model.trained_model = clf
                        model.grid_param_dict = grid_param_dict
                        model.x_columns = x_columns
                        for k, v in grid.cv_results_.items():
                            if isinstance(v, (list)) and all([isinstance(v1, (float,complex,int,Number)) for v1 in v]):
                                v = mean(v)
                            print('{0}\t: {1}'.format(k, v))
                    break

                except (ValueError) as e:
                    if 'Invalid parameter ' in str(e):
                        grid_param_dict.pop(str(e).split(' ')[2])
                        pass
                    else:
                        raise e
        else:
            if retrain_model and \
                    (model.trained_model is not None) and \
                    (model.x_columns == x_columns) and \
                    (clf.get_params() == model.trained_model.get_params()):
                clf = model.trained_model
            else:
                clf, x_train = self.resilient_fit(clf, x_train, y_train)
                model.trained_model = clf
                model.x_columns = x_columns

        print("------ Model Training Complete [{0}] ------\n".format(model.model_class))

        return model, x_train


    def train_models(self,
                     df: pd.DataFrame,
                     field_names: dict
                     ):
        trained_models = dict()
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
                vectorizer = sk_text.TfidfVectorizer(stop_words='english', analyzer='word', lowercase=True, min_df=0.001)
                vectorizer.fit(filter(None, map(str.strip, df[f][~df[f].isnull()].values.astype(str))))
                # print("Feature Names for TF/IDF Vectorizer\n", *[v.encode('ascii', errors='ignore')
                #       .decode('utf-8', errors='ignore') for v in vectorizer.get_feature_names()])
    
            elif t == 'num':
                if self.use_sparse:
                    vectorizer = sk_prep.MaxAbsScaler()
                    # vectorizer = sk_prep.StandardScaler(with_mean=False)
                else:
                    vectorizer = sk_prep.StandardScaler()
                    # vectorizer = sk_prep.MinMaxScaler()
                vectorizer.fit(df[f].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64).reshape(-1, 1))
    
            elif t == 'num_noscale':
                vectorizer = None
    
            else:
                raise ValueError('Invalid column type provided. Choose between: \'num\', \'cat\', and \'doc\'.')
            trained_models[f] = vectorizer
    
        return trained_models
    
    
    def get_vectors(self,
                    df: pd.DataFrame,
                    field_names: dict,
                    trained_models: dict,
                    is_y: bool=False
                    ):
    
        final_matrix = None
        column_names = list()
        for f, t in list(field_names.items()):
            if self.verbose:
                print("TRANSFORMING: ", f, end='')
            df[f + '_isnull'] = df[f].isnull().values.astype(np.float64).reshape(-1, 1)
            if self.use_sparse:
                null_matrix = sparse.csc_matrix(df[f + '_isnull'], dtype=np.float64)
            else:
                null_matrix = df[f + '_isnull'].as_matrix().astype(np.float64)
    
            # print(trained_models[f])
            if isinstance(trained_models[f], sk_prep.LabelEncoder):
                while 1:
                    try:
                        matrix = trained_models[f].transform(df[f].values.astype(str).transpose())
                        break
                    except ValueError as e:
                        print("\nAll Uniques in Series:", sorted(df[f].unique().astype(str)))
                        print('Length: %s' % len(df[f].unique().astype(str)))
    
                        # print(str(e).split("\'"))
                        # for k, v in enumerate(str(e).split("\'")):
                        #     if k % 2 == 1:
                        #         trained_models[f].classes_ = np.append(trained_models[f].classes_, v)
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
                        print('Added new classes to encoder: %s ' % new_classes)
                        print('All Classes: %s' % sorted(trained_models[f].classes_))
                        print('Length: %s' % len(trained_models[f].classes_))
                        pass
    
            elif t == 'cat':
                mkdict = lambda row: dict((col, row[col]) for col in [f])
                matrix = trained_models[f].transform(df.apply(mkdict, axis=1)).transpose()
                for v in trained_models[f].get_feature_names():
                    column_names.append((v, True, f))
    
            elif t == 'doc':
                matrix = trained_models[f].transform(df[f].values.astype(str)).transpose()
                # for i in range(1, matrix.shape[0]):
                #     column_names.append((f, False))
                for v in trained_models[f].get_feature_names():
                    # Change to false if you want to eliminate the (MANY) possible words
                    column_names.append((v.encode('ascii', errors='ignore').decode('utf-8', errors='ignore'), True, f))
                column_names.append((f + '_isnull', True, f))
    
                if self.use_sparse:
                    matrix = sparse.csc_matrix(matrix)
                matrix = self.matrix_vstack((matrix, null_matrix))
    
            # else:
            elif t == 'num':
                df[f + '_t'] = trained_models[f].transform(df[f]
                                .apply(pd.to_numeric, errors='coerce')
                                .round(4).fillna(0).values.reshape(-1,1))
                if is_y:
                    matrix = df[f + '_t'].astype(np.float64).values.transpose()
    
                    if isinstance(trained_models[f], sk_prep.MinMaxScaler):
                        matrix = np.minimum(1, np.maximum(0, fix_np_nan(matrix)))
    
                        # matrix.data = np.minimum(1, np.maximum(0, fix_np_nan(matrix)))
                else:
                    matrix = df[f + '_t'].astype(np.float64).values
                    if isinstance(trained_models[f], sk_prep.MinMaxScaler):
                        matrix = np.minimum(1, np.maximum(0, fix_np_nan(matrix)))
    
                    if self.use_sparse:
                        matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    
                    # matrix = sparse.vstack((matrix, null_matrix))
                    matrix = self.matrix_vstack((matrix, null_matrix))
    
                    # matrix.data = np.minimum(1, np.maximum(0, np.nan_to_num(matrix.data)))
                    column_names.append((f + '_t', True, f))
                    column_names.append((f + '_isnull', True, f))
    
            elif t == 'num_noscale':
                matrix = df[f].astype(np.float64).values.transpose()
                if self.use_sparse:
                    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    
                matrix = self.matrix_vstack((matrix, null_matrix))
    
                # matrix.data = np.minimum(1, np.maximum(0, np.nan_to_num(matrix.data)))
                column_names.append((f + '_t', True, f))
                column_names.append((f + '_isnull', True, f))
    
            if self.verbose:
                print(' || Shape: {0}'.format(matrix.shape))
            if final_matrix is not None:
                if is_y:
                    final_matrix.append(matrix)
                else:
                    final_matrix = self.matrix_vstack((final_matrix, matrix))
    
            else:
                final_matrix = matrix
        else:
            if not is_y:
                if self.use_sparse:
                    final_matrix = sparse.csr_matrix(final_matrix)
                elif len(final_matrix.shape) == 1:
                    final_matrix.reshape(-1, 1)
                final_matrix = final_matrix.transpose()
    
            return column_names, final_matrix, trained_models
    
    
    def matrix_vstack(self, m: tuple, return_sparse: bool=None):
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
    
    
    def matrix_hstack(self, m: tuple, return_sparse: bool=None):
        if sum([sparse.issparse(d) for d in list(m)]) == 1:
            if self.use_sparse:
                m = [sparse.csc_matrix(d) if not sparse.issparse(d) else d for d in list(m)]
            else:
                m = [d.toarray() if sparse.issparse(d) else (d.reshape(-1,1) if len(d.shape)==1 else d) for d in list(m)]
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


def fix_np_nan(m: Union[np.matrix, sparse.csr_matrix]) -> Union[np.matrix, sparse.csr_matrix]:
    if sparse.issparse(m):
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


def reverse_enumerate(l):
    for index in reversed(range(len(l))):
        yield index, l[index]


def get_decimals(vals: np.ndarray):
    max_decimals = max([len(v.split('.')[-1]) for v in vals if (not np.isnan(v)) and '.' in v])
    # l = [v.split('.')[-1]]
    # # x = str(x).rstrip('0')  # returns '56.001'
    # x = decimal.Decimal(x)  # returns Decimal('0.001')
    # x = x.as_tuple().exponent  # returns -3
    # x = abs(x)
    return max_decimals