__author__ = 'Andrew Mackenzie'
import os
import sys

if os.path.abspath(os.pardir) not in sys.path:
    sys.path.append(os.path.abspath(os.pardir))

from typing import Union
from math import exp
import math
from Settings import Settings
from collections import OrderedDict
import numpy as np
import prettytable
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
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.linear_model.logistic import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression, RidgeClassifier, RidgeClassifierCV, BayesianRidge, RidgeCV
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC, SVR, NuSVR, NuSVC
from sklearn.metrics import r2_score
from scipy.special import expit
from sklearn import metrics
import warnings

pd.options.mode.chained_assignment = None

global use_sparse, verbose, s, new_fields
use_sparse = True

def predict(df: pd.DataFrame
            , x_fields: dict
            , y_field: Union[dict, OrderedDict]
            , model_type: Union[str, list]
            , report_name: str
            , show_model_tests: bool=False
            , retrain_model: bool=False
            , selection_limit: float=5.0e-2
            , predict_all: bool=False
            , verbose: bool=True
            , train_pct: float=0.7
            , random_train_test: bool=True
            , max_train_size: int=50000
            ) -> pd.DataFrame:

    global s
    globals()['verbose'] = verbose
    s = Settings(report_name=report_name)
    final_file_dir = s.get_default_data_dir()
    # os.makedirs(final_file_dir, exist_ok=True)

    if not verbose:
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=exceptions.ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    model_dir = s.get_model_dir()
    # os.makedirs(model_dir, exist_ok=True)

    if not isinstance(model_type, list):
        model_type = [model_type]

    predictive_model_file = '{0}\predictive_model_{1}.p'\
        .format(model_dir,
                model_type[0] if len(model_type) == 1 else 'stacked'
                )

    mappings_file = '%s\%s' % (model_dir, 'data_mappings.p')

    for k in list(y_field.keys()):
        df[k].replace('', np.nan, inplace=True)
        # print("Value Counts in dataframe\n", df[k].value_counts())
        if predict_all:
            mask_validate = np.asarray([True for v in range(df.shape[0])], dtype=np.bool)
        else:
            mask_validate = df[k].isnull()
        mask_train_test = ~df[k].isnull()
        # print('Final Status field\n', df[k])
        # print('Value Counts\n', mask_train_test.value_counts())
        # print('DF Length\n', len(df[k]))
        # print('Validate Dataset\n', mask_train_test.value_counts().loc[True])

        mask_train_test_qty_true = mask_train_test.value_counts().loc[True]

        train_pct = min(train_pct, max_train_size / (len(df[k]) - mask_train_test_qty_true))
        test_pct = min((1-train_pct), max_train_size / (len(df[k]) - mask_train_test_qty_true))

        if verbose:
            print("Train data fraction:", train_pct)
            print("Test data fraction:", test_pct)

        if random_train_test:
            mask = np.random.rand(len(df[k]))
            mask_train = [a and b for a, b in zip(mask <= train_pct, mask_train_test != False)]
            mask_test = [a and b for a, b in zip(mask > (1-test_pct), mask_train_test != False)]
        else:
            mask_train = mask_train_test.copy()
            mask_test = mask_train_test.copy()
            mask_train_qty = 0
            for idx, val in enumerate(mask_train_test):
                if val:
                    if float(mask_train_qty) <= mask_train_test_qty_true * train_pct:
                        mask_test[idx] = False
                    else:
                        mask_train[idx] = False
                    mask_train_qty += 1
        # print("Mask_Train Value Counts\n", pd.Series(mask_train).value_counts())
        # print("Mask_Test Value Counts\n", pd.Series(mask_test).value_counts())
    # x_mappings, x_columns, x_vectors = set_vectors(df.loc[mask_train_test,], x_fields)
    # y_mappings, y_vectors = set_vectors(df, y_field, is_y=True)

    try:
        if retrain_model:
            raise ValueError('"retrain_model" variable is set to True. Training new preprocessing & predictive models.')
        with open(mappings_file, 'rb') as pickle_file:
            x_mappings, y_mappings = pickle.load(pickle_file)
        with open(predictive_model_file, 'rb') as pickle_file:
            clf, = pickle.load(pickle_file)
        print('Successfully loaded the mappings and predictive model.')
    except (FileNotFoundError, ValueError) as e:
        print(e)

        # CHANGE THE 'cat' to a list of all possible values in the y field. This is necessary because the LabelEncoder
        # that will encode the Y variable values can't handle never-before-seen values. So we need to pass it every
        # possible value in the Y variable, regardless of whether it appears in the train or test subsets.
        for k, v in y_field.items():
            if v == 'cat':
                y_field[k] = df[k].unique().astype(str)

        data_rows = len(mask_train)
        y_mappings = train_models(df[mask_train], y_field)
        _, y_train, y_mappings = get_vectors(df[mask_train], y_field, y_mappings, is_y=True)

        x_mappings = train_models(df[mask_train], x_fields)
        x_columns, x_train, x_mappings = get_vectors(df[mask_train], x_fields, x_mappings)
        x_train = fix_np_nan(x_train)

        for idx, mt in enumerate(model_type):

            # print(x_train.shape)
            if verbose:
                print('FIELDS\n', np.asarray(list(x_fields.keys())))
            # y_train, uniques_index = pd.factorize(train[y_fields])

            if not isinstance(mt, str) and hasattr(mt, 'fit'):
                clf = mt
                mt = 'custom'
            elif mt == 'rfor':
                clf = RandomForestClassifier(n_estimators=31)
            elif mt == 'rfor_r':
                clf = RandomForestRegressor(n_estimators=31)
            elif mt == 'logit':
                clf = LogisticRegressionCV()
            elif mt == 'linreg':
                clf = LinearRegression()
            elif mt == 'ridge':
                clf = RidgeCV()
            elif mt == 'lasso':
                clf = Lasso()
            elif mt == 'elastic_net':
                clf = ElasticNet()
            elif mt == 'elastic_net_stacking':
                clf = ElasticNet(positive=True)  # Used positive=True to make this ideal for stacking
            elif mt == 'neural_c':
                clf = MLPClassifier(learning_rate='adaptive', learning_rate_init=0.1)
            elif mt == 'neural_r':
                clf = MLPRegressor(learning_rate='adaptive', learning_rate_init=0.1)
            elif mt == 'svc':
                clf = SVC()
            elif mt == 'svr':
                clf = SVR()
            elif mt == 'nu_svc':
                clf = NuSVC()
            elif mt == 'nu_svr':
                clf = NuSVR()
            elif mt == 'gbc':
                clf = GradientBoostingClassifier(n_estimators=int(round(x_train.shape[0]/20, 0)))
            elif mt == 'gbr':
                clf = GradientBoostingRegressor(n_estimators=int(round(x_train.shape[0]/20, 0)))
            elif mt == 'abc':
                clf = AdaBoostClassifier(n_estimators=int(round(x_train.shape[0]/20, 0)))
            elif mt == 'knn_c':
                clf = KNeighborsClassifier()
            elif mt == 'knn_r':
                clf = KNeighborsRegressor()
            elif mt == 'ridge_c':
                clf = RidgeClassifierCV()
            elif mt == 'linear_svc':
                clf = LinearSVC()
            elif mt == 'sgd_c':
                clf = SGDClassifier()
            elif mt == 'sgd_r':
                clf = SGDRegressor()
            elif mt == 'pass_agg_c':
                clf = PassiveAggressiveClassifier()
            elif mt == 'pass_agg_r':
                clf = PassiveAggressiveRegressor()
            elif mt == 'bernoulli_nb':
                clf = BernoulliNB()
            elif mt == 'bernoulli_rbm':
                clf = BernoulliRBM()
            elif mt == 'multinomial_nb':
                clf = MultinomialNB()
            elif mt == 'nearest_centroid':
                clf = NearestCentroid()
            else:
                raise ValueError('Incorrect model_type given. Cannot match [%s] to a model.' % mt)

            global use_sparse
            if isinstance(clf, (GradientBoostingRegressor, KNeighborsClassifier, KNeighborsRegressor, SVR)):
                use_sparse = False
                if not isinstance(x_train, (np.ndarray)):
                    x_train = x_train.toarray()
                if not isinstance(y_train, (np.ndarray)):
                    y_train = y_train.toarray().reshape(-1, 1)
            else:
                use_sparse = True

            if 'random_state' in clf.get_params().keys():
                clf.random_state = 555
            if 'verbose' in clf.get_params().keys():
                clf.verbose = verbose
            if 'class_weight' in clf.get_params().keys():
                clf.class_weight = 'balanced'
            if 'probability' in clf.get_params().keys():
                clf.probability = True
            if 'max_iter' in clf.get_params().keys():
                if (not clf.max_iter) or (clf.max_iter < 10000):
                    clf.max_iter = 10000
            # if 'learning_rate' in clf.get_params().keys():
            #     clf.learning_rate = 'adaptive'
            # if 'learning_rate_init' in clf.get_params().keys():
            #     clf.learning_rate_init = 0.1

            mask_all = np.asarray([True for v in range(df.shape[0])], dtype=np.bool)
            # print("NaN in x_train: %s" % np.isnan(x_train.data).any())
            # print("NaN in y_train: %s" % np.isnan(y_train.data).any())

            print("\n----- Training Predictive Model -----")

            if mt != 'custom':
                grid_param_dict = dict()
                # ALPHAS
                if 'alpha' in clf.get_params().keys():
                    if isinstance(clf, (GradientBoostingRegressor)):
                        vals = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
                    else:
                        vals = [100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
                    # if hasattr(clf, 'learning_rate') and clf.learning_rate == 'optimal':
                    #     del vals[-1]
                    grid_param_dict['alpha'] = vals

                # TOLS
                if 'tol' in clf.get_params().keys():
                    grid_param_dict['tol'] = [1000000, 100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

                # TOLS
                if 'n_neighbors' in clf.get_params().keys():
                    grid_param_dict['n_neighbors'] = [2, 3, 4, 5, 6, 7, 8, 9]

                # SELECTION
                if 'selection' in clf.get_params().keys():
                    grid_param_dict['selection'] = ['cyclic','random']

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
                    if isinstance(clf, (SVC, SVR)):
                        grid_param_dict['kernel'] = ['linear','poly','rbf','sigmoid']  # LINEAR IS 'Work in progress.' as of 0.19
                    else:
                        print('Unspecified parameter "kernel" for ', type(clf))

                    # NU
                    if 'nu' in clf.get_params().keys():
                        grid_param_dict['nu'] = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

                # METRICS
                if 'metric' in clf.get_params().keys():
                    if isinstance(clf, (NearestCentroid)):
                        grid_param_dict['metric'] = ['euclidean', 'manhattan']
                    if isinstance(clf, (KNeighborsRegressor, KNeighborsClassifier)):
                        grid_param_dict['metric'] = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
                        # NOTE: wminkowski doesn't seem to work in here. Might be a bug in sklearn 0.19.
                        # NOTE: I deliberately left out seuclidean and mahalanobis because they were too much trouble.
                    else:
                        print('Unspecified parameter "metric" for ', type(clf))

                while True:
                    try:
                        grid = GridSearchCV(estimator=clf, param_grid=grid_param_dict, n_jobs=-1)
                        grid.fit(x_train, y_train)
                        break
                    except ValueError as e:
                        if 'Invalid parameter ' in str(e):
                            grid_param_dict.pop(str(e).split(' ')[2])
                        else:
                            raise e
                        pass
                print(grid)
                # summarize the results of the grid search
                print('Grid Regression Best Score:', grid.best_score_)
                print('Grid Regression Best Estimator:', grid.best_estimator_)
                clf = grid.best_estimator_

            elif selection_limit < 1.0:
                scores, p_vals = sk_feat_sel.f_regression(x_train, y_train, center=False)
                for x_field_name in list(x_fields.keys()):
                    xcol_indices = [idx for idx, vals in enumerate(x_columns) if vals[2] == x_field_name]
                    if all(p_vals[idx] > selection_limit or p_vals[idx] == np.nan for idx in xcol_indices):
                        x_fields.pop(x_field_name)

                x_columns, x_train, x_mappings = get_vectors(df[mask_train], x_fields, x_mappings)

                x_train = fix_np_nan(x_train)

            clf.fit(x_train, y_train)
            print("------ Model Training Complete ------\n")


            # Loop until you reach the very last predictive model, iteratively adding the predictions from each model
            #  to the dataframe and including those predictions in each successive model training process
            if idx < len(model_type)-1:
                max_slice_size = 100000
                y_all = list()
                # y_all_probs = list()
                for s in range(0, int(math.ceil(len(df.index) / max_slice_size))):
                    min_idx = s * max_slice_size
                    max_idx = min(len(df.index), (s + 1) * max_slice_size)
                    print("Prediction Iteration #%s: min/max = %s/%s" % (s, min_idx, max_idx))

                    df_valid_iter = df[mask_all].iloc[min_idx:max_idx]
                    x_valid_columns, x_all, x_mappings = get_vectors(df_valid_iter, x_fields, x_mappings)
                    x_all = fix_np_nan(x_all)

                    # print(x_validate)

                    try:
                        preds = clf.predict(x_all)
                        # if hasattr(clf, 'classes_'):
                        #     pred_probs = clf.predict_proba(x_all)
                    except TypeError as e:
                        if "dense data is required" in str(e):
                            x_all = x_all.toarray()
                            use_sparse = False
                            preds = clf.predict(x_all)
                            pass
                    except DeprecationWarning as e:
                        print("YOU NEED TO FIX THE BELOW ERROR SINCE IT WILL BE DEPRECATED")
                        print(e)
                        x_all = x_all.reshape(-1, 1)
                        preds = clf.predict(x_all)
                        # if hasattr(clf, 'classes_'):
                        #     pred_probs = clf.predict_proba(x_all)
                        pass
                    finally:
                        y_all.append(preds)

                # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
                y_all = np.hstack(y_all).reshape(-1,1)

                # Generate predictions for this predictive model, for all values. Add them to the DF so they can be
                #  used as predictor variables (e.g. "stacked" upon) later when the next predictive model is run.
                for k, v in y_field.items():
                    # y_all = np.atleast_1d(y_all)
                    preds = y_mappings[k].inverse_transform(y_all)
                    print('INTERMEDIATE PREDICTIONS')
                    print(*preds[:10])
                    pred_x_name = '{0}_pred_{1}'.format(mt, k)
                    df.loc[:, pred_x_name] = preds
                    new_x_fields = {pred_x_name: 'cat' if not isinstance(v, str) else v}
                    x_fields = {**x_fields, **new_x_fields}
                    x_mappings = {**x_mappings, **train_models(df[mask_train], new_x_fields)}
                    new_x_columns, new_x_train, x_mappings = get_vectors(df[mask_train], new_x_fields, x_mappings)
                    x_columns += new_x_columns
                    new_x_train = fix_np_nan(new_x_train)
                    # new_x_train = sparse.csc_matrix(np.hstack(new_x_train))
                    x_train = matrix_hstack((x_train, new_x_train), return_sparse=True)

        try:
            with open(predictive_model_file, 'wb') as pickle_file:
                pickle.dump((clf,), pickle_file)
        except AttributeError:
            pass

        with open(mappings_file, 'wb') as pickle_file:
            pickle.dump((x_mappings, y_mappings), pickle_file)

    if show_model_tests:
        x_columns, x_test, x_mappings = get_vectors(df[mask_test], x_fields, x_mappings)
        _, y_test, y_mappings = get_vectors(df[mask_test], y_field, y_mappings, is_y=True)
        x_test = fix_np_nan(x_test)

        print("Total Named Columns: ", len(x_columns))

        coef_list = [['name'] + [new_colname.strip() for new_colname, show, orig_colname in x_columns]]

        if hasattr(clf, 'feature_importances_'):
            feat_importances = ['coef'] + fix_np_nan(clf.feature_importances_).tolist()
            print('Feature Importances:\n%s' % clf.feature_importances_, sep='\n')
            coef_list.append(feat_importances)

        try:
            coefs = []
            if hasattr(clf, 'coef_'):
                if isinstance(clf.coef_[0], (tuple, list, np.ndarray)):
                    coefs = fix_np_nan([v[0] for v in clf.coef_])
                else:
                    coefs = fix_np_nan([v for v in clf.coef_])
                if isinstance(coefs, (np.ndarray)) :
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

            scores, p_vals = sk_feat_sel.f_regression(x_test, y_test, center=False)

            nonzero_coefs_mask = [v != 0 for v in coefs]
            significant_coefs_mask = [v <= selection_limit for v in p_vals]

            print_df = pd.DataFrame(list(zip([v[0] for v in x_columns]
                                            , coefs
                                            , scores.tolist()
                                            , p_vals.tolist()))
                                    , columns=['Name', 'Coef', 'Score', 'P-Value'])

            print_df = print_df.loc[[v1 and v2 for v1, v2 in zip(nonzero_coefs_mask, significant_coefs_mask)], :]
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
                coefs = ['coefs'] + [v for v in coefs]
                coef_list.append(coefs)

                expit_coefs = ['coef_expit'] + [expit(float(v)) for v in coefs[1:]]
                coef_list.append(expit_coefs)

                exp_coefs = ['coef_exp'] + ['%f' % exp(float(v)) for v in coefs[1:]]
                coef_list.append(exp_coefs)

        except ValueError as e:
            print(e)
            pass

        # scores, pvalues = chi2(x_test, y_test)
        scores2, pvalues2 = sk_feat_sel.f_regression(x_test, y_test, center=False)
        scores2 = ['linreg_scores'] + scores2.tolist()
        pvalues2 = ['linreg_pvalues'] + pvalues2.tolist()
        coef_list.append(scores2)
        coef_list.append(pvalues2)

        scores3, pvalues3 = sk_feat_sel.f_classif(x_test, y_test)
        scores3 = ['classif_scores'] + scores3.tolist()
        pvalues3 = ['classif_pvalues'] + pvalues3.tolist()
        coef_list.append(scores3)
        coef_list.append(pvalues3)

        coef_list = list(map(list, zip(*coef_list)))
        # coef_list = coef_list.transpose()
        for i, r in enumerate(coef_list[1:]):
            coef_list[i+1] = [r[0]] + [round(float(v),4) for v in r[1:]]

        value_mask = [bool(v[1]) for v in x_columns]
        value_mask_95 = [bool(x_columns[i][1]) and (float(v) <= 0.05) for i, v in enumerate(pvalues2[1:])]

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
            all_coefs.to_csv('%s/%s' % (final_file_dir, 'all_coefs.csv'), index=True, index_label=['index1'], header=True, encoding='utf_8')
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
        try:
            preds = clf.predict(x_test)
        except TypeError as e:
            if "dense data is required" in str(e):
                x_test = x_test.toarray()
                use_sparse = False
                preds = clf.predict(x_test)
        print('R2 SCORE\n', r2_score(y_test, preds))

        if hasattr(clf, 'classes_'):
            print('CLASSES')
            print(clf.classes_)
            print('ACCURACY:\n', metrics.accuracy_score(y_test, preds))
            try:
                y_test_probs = clf.predict_proba(x_test)
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

            for k, v in y_field.items():
                preds_str = resilient_inverse_transform(y_mappings[k], preds)
                preds_names = list(set(preds_str))

                print("Unique integers in y_test:", list(set(y_test)))
                y_test_str = resilient_inverse_transform(y_mappings[k], y_test)
                y_test_names = list(set(y_test_str))
                # Create confusion matrix
                conf_matrix = pd.crosstab(y_test_str, preds_str, rownames=['actual'],
                                          colnames=['predicted'])
                print("CONFIDENCE MATRIX")
                print(conf_matrix)

        pass

    # CREATE X_VALIDATE
    df_validate = df.loc[mask_validate, :]

    # ITERATE OVER THE df_validate in groups of 100K rows (to avoid memory errors) and predict outcomes
    max_slice_size = 100000
    y_validate = list()
    y_validate_probs = list()
    for s in range(0, int(math.ceil(len(df_validate.index)/max_slice_size))):
        min_idx = s*max_slice_size
        max_idx = min(len(df_validate.index), (s+1)*max_slice_size)
        print("Prediction Iteration #%s: min/max = %s/%s" % (s, min_idx, max_idx))

        df_valid_iter = df[mask_validate].iloc[min_idx:max_idx]
        x_valid_columns, x_validate, x_mappings = get_vectors(df_valid_iter, x_fields, x_mappings)

        x_validate = fix_np_nan(x_validate)

        # print(x_validate)
        try:
            preds = clf.predict(x_validate)
        except TypeError as e:
            if "dense data is required" in str(e):
                x_validate = x_validate.toarray()
                use_sparse = False
                preds = clf.predict(x_validate)
                pass
        except DeprecationWarning as e:
            print("YOU NEED TO FIX THE BELOW ERROR SINCE IT WILL BE DEPRECATED")
            print(e)
            x_validate = x_validate.reshape(-1, 1)
            preds = clf.predict(x_validate)
            pass

        calculate_probs = hasattr(clf, 'classes_') \
                          and hasattr(clf, 'predict_proba') \
                          and not (hasattr(clf, 'loss') and clf.loss == 'hinge')
        if calculate_probs:
            pred_probs = clf.predict_proba(x_validate)
            y_validate_probs.append(pred_probs)

        y_validate.append(preds)

    # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
    y_validate = np.hstack(y_validate).reshape(-1,1)

    global new_fields
    new_fields = list()
    if calculate_probs:
        y_validate_probs = np.vstack(y_validate_probs)
        print('ORIGINAL PROBABILITIES')
        print(*y_validate_probs[:10].round(4), sep='\n')
        print('ORIGINAL PROBABILITIES (NORMALIZED)')
        print(*[np.around(np.expm1(x), 4) for x in y_validate_probs][:10], sep='\n')

        for y_name, m in y_mappings.items():
            df_probs = pd.DataFrame(data=y_validate_probs.round(4)
                            , columns=[y_name + '_prob_' + f.lower().replace(' ','_') for f in list(m.classes_)[:y_validate_probs.shape[1]]]
                            , index=df_validate.index)
            # df_validate.reset_index(inplace=True, drop=True)
            df_validate = df_validate.join(df_probs, how='inner')
            # for col in df_probs.columns.values:
            #     df_validate[col] = df_probs[col]
        new_fields.extend(df_probs.columns.values.tolist())

    for k, v in y_field.items():
        final_preds = y_mappings[k].inverse_transform(y_validate)
        print('FINAL PREDICTIONS')
        print(*final_preds[:10])
        y_pred_name = 'pred_' + k
        df_validate.loc[:, y_pred_name] = final_preds
        new_fields.extend([y_pred_name])

    if predict_all:
        df = df_validate
    else:
        df = df.loc[mask_validate == 0, :].append(df_validate, ignore_index=True)
        df = df.loc[mask_validate == 0, :].append(df_validate, ignore_index=True)

    return df

def get_pred_field_names():
    global new_fields
    return new_fields

def resilient_inverse_transform(model: sk_prep.LabelEncoder, preds: np.ndarray):
    try:
        preds_str = model.inverse_transform(preds)
    except TypeError as e:  # This will handle a bug with using a Numpy ndarray as an index. FUCKING HELL.
        print(e)
        preds = preds.tolist()
        preds_str = [model.classes_[i] for i in preds]
        pass
    return preds_str


def fix_np_nan(m: Union[np.matrix, sparse.csr_matrix]) -> Union[np.matrix, sparse.csr_matrix]:
    if sparse.issparse(m):
        m.data = np.nan_to_num(m.data)
    else:
        m = np.nan_to_num(m)
    return m


def is_number(s) -> bool:
    try:
        float(s) # for int, long and float
    except ValueError:
        return False
    return True


def train_models(df: pd.DataFrame
                 , field_names: dict
                 ):
    trained_models = dict()
    for f, t in list(field_names.items()):
        if verbose:
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
            if use_sparse:
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


def get_vectors(df: pd.DataFrame
                , field_names: dict
                , trained_models: dict
                , is_y: bool=False
                ):

    final_matrix = None
    column_names = list()
    for f, t in list(field_names.items()):
        if verbose:
            print("TRANSFORMING: ", f, end='')
        df[f + '_isnull'] = df[f].isnull().values.astype(np.float64).reshape(-1, 1)
        if use_sparse:
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

            if use_sparse:
                matrix = sparse.csc_matrix(matrix)
            matrix = matrix_vstack((matrix, null_matrix))

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

                if use_sparse:
                    matrix = sparse.csc_matrix(matrix, dtype=np.float64)

                # matrix = sparse.vstack((matrix, null_matrix))
                matrix = matrix_vstack((matrix, null_matrix))

                # matrix.data = np.minimum(1, np.maximum(0, np.nan_to_num(matrix.data)))
                column_names.append((f + '_t', True, f))
                column_names.append((f + '_isnull', True, f))

        elif t == 'num_noscale':
            matrix = df[f].astype(np.float64).values.transpose()
            if use_sparse:
                matrix = sparse.csc_matrix(matrix, dtype=np.float64)

            matrix = matrix_vstack((matrix, null_matrix))

            # matrix.data = np.minimum(1, np.maximum(0, np.nan_to_num(matrix.data)))
            column_names.append((f + '_t', True, f))
            column_names.append((f + '_isnull', True, f))

        if verbose:
            print(' || Shape: {0}'.format(matrix.shape))
        if final_matrix is not None:
            if is_y:
                final_matrix.append(matrix)
            else:
                final_matrix = matrix_vstack((final_matrix, matrix))

        else:
            final_matrix = matrix
    else:
        if not is_y:
            if use_sparse:
                final_matrix = sparse.csr_matrix(final_matrix)
            elif len(final_matrix.shape) == 1:
                final_matrix.reshape(-1, 1)
            final_matrix = final_matrix.transpose()

        return column_names, final_matrix, trained_models


def matrix_vstack(m: tuple, return_sparse: bool=None):
    global use_sparse
    if sum([sparse.issparse(d) for d in list(m)]) == 1:
        use_sparse = False
        m = [d.toarray() if sparse.issparse(d) else d for d in list(m)]
        m = tuple(m)

    if use_sparse:
        m = sparse.vstack(m)
        if return_sparse is False:
            m = m.toarray()
    else:
        m = np.vstack(m)
        if return_sparse:
            m = sparse.csc_matrix(m)
    return m


def matrix_hstack(m: tuple, return_sparse: bool=None):
    global use_sparse
    if sum([sparse.issparse(d) for d in list(m)]) == 1:
        use_sparse = False
        m = [d.toarray() if sparse.issparse(d) else (d.reshape(-1,1) if len(d.shape)==1 else d) for d in list(m)]
        m = tuple(m)

    if use_sparse:
        m = sparse.hstack(m)
        if return_sparse is False:
            m = m.toarray()
    else:
        m = np.hstack(m)
        if return_sparse:
            m = sparse.csc_matrix(m)
    return m


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