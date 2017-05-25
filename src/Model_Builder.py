__author__ = 'Andrew Mackenzie'
import os
import sys

if os.path.abspath(os.pardir) not in sys.path:
    sys.path.append(os.path.abspath(os.pardir))

from typing import Union
from math import exp
import math
import Settings
from collections import OrderedDict
import numpy as np
import prettytable
import pandas as pd
import pickle
import bisect
from scipy import sparse
import sklearn.feature_extraction.text as sk_text
import sklearn.preprocessing as sk_prep
from sklearn import feature_extraction as sk_feat
from sklearn import feature_selection as sk_feat_sel
from sklearn.neural_network import MLPClassifier as cmlp
from sklearn.neural_network import MLPRegressor as rmlp
from sklearn.linear_model import Ridge as ridge
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model.logistic import LogisticRegression as logr
from sklearn.ensemble import RandomForestClassifier as rfor
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.linear_model import LinearRegression as linreg
from sklearn.svm import SVC as svc
from sklearn.svm import SVR as svr
from sklearn.metrics import r2_score
from scipy.special import expit
from sklearn import metrics

pd.options.mode.chained_assignment = None

global use_sparse, verbose
use_sparse=True

def predict(df: pd.DataFrame
            , x_fields: OrderedDict
            , y_field: OrderedDict
            , model_type: str
            , report_name: str
            , show_model_tests: bool=False
            , retrain_model: bool=False
            , selection_limit: float=5.0e-2
            , predict_all: bool=False
            , verbose: bool=True
            , train_pct: float=0.7
            , random_train_test: bool=True
            ) -> pd.DataFrame:


    globals()['verbose'] = verbose
    reportName = report_name
    final_file_dir = Settings.default_base_dir + '/%s%s' % (reportName, Settings.default_data_dir)
    os.makedirs(final_file_dir, exist_ok=True)

    model_dir = '%s/%s/' % (Settings.default_model_dir, reportName)
    os.makedirs(model_dir, exist_ok=True)

    if not isinstance(model_type, str) and hasattr(model_type, 'fit'):
        clf = model_type
        model_type = 'custom'
    predictive_model_file = '%s%s' % (model_dir, 'predictive_model_%s.p' % model_type)

    mappings_file = '%s%s' % (model_dir, 'data_mappings.p')

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

        train_pct = min(train_pct, 50000 / (len(df[k]) - mask_train_test_qty_true))
        test_pct = min((1-train_pct), 50000 / (len(df[k]) - mask_train_test_qty_true))

        if verbose:
            print("Train data fraction:", train_pct)
            print("Test data fraction:", test_pct)

        if random_train_test:
            mask = np.random.rand(len(df[k]))
            mask_train = [a and b for a, b in zip(mask <= train_pct, mask_train_test == 0)]
            mask_test = [a and b for a, b in zip(mask > (1-test_pct), mask_train_test == 0)]
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
            (x_mappings, y_mappings) = pickle.load(pickle_file)
        with open(predictive_model_file, 'rb') as pickle_file:
            (clf,) = pickle.load(pickle_file)
        print('Successfully loaded the mappings and predictive model.')
    except (FileNotFoundError, ValueError) as e:
        print(e)

        # CHANGE THE 'cat' to a list of all possible values in the y field. This is necessary because the LabelEncoder
        # that will encode the Y variable values can't handle never-before-seen values. So we need to pass it every
        # possible value in the Y variable, regardless of whether it appears in the train or test subsets.
        for k, v in y_field.items():
            if v == 'cat':
                y_field[k] = df[k].unique().astype(str)

        x_mappings = train_models(df[mask_train], x_fields)
        y_mappings = train_models(df[mask_train], y_field)
        with open(mappings_file, 'wb') as pickle_file:
            pickle.dump((x_mappings, y_mappings), pickle_file)

        x_columns, x_train, x_mappings = get_vectors(df[mask_train], x_fields, x_mappings)
        y_train, y_mappings = get_vectors(df[mask_train], y_field, y_mappings, is_y=True)

        # print(x_train.shape)
        if verbose:
            print('FIELDS\n', np.asarray(list(x_fields.keys())))
        # y_train, uniques_index = pd.factorize(train[y_fields])
        global use_sparse
        if model_type == 'custom':
            clf = clf
            if isinstance(clf, (gbr)):
                use_sparse = False
        elif model_type == 'rfor':
            clf = rfor(random_state=555, verbose=True, n_estimators=31)
        elif model_type == 'logit':
            clf = logr(random_state=555, verbose=True)
        elif model_type == 'linreg':
            clf = linreg()
        elif model_type == 'ridge':
            clf = ridge(random_state=555)
        elif model_type == 'neural_c':
            clf = cmlp(random_state=555, verbose=True, learning_rate_init=0.1, learning_rate='adaptive'
                       # , max_iter=int(round(x_train.shape[0]/2000, 0))
                       )
        elif model_type == 'neural_r':
            clf = rmlp(random_state=555, verbose=True, learning_rate_init=0.1, learning_rate='adaptive'
                       # , max_iter=int(round(x_train.shape[0]/2000, 0))
                       )
        elif model_type == 'svc':
            clf = svc(random_state=555, verbose=True, kernel='rbf', probability=True, max_iter=1000)
        elif model_type == 'svr_lin':
            clf = svr(verbose=True, kernel='linear', max_iter=1000)
        elif model_type == 'svr_rbf':
            clf = svr(verbose=True, kernel='rbf', max_iter=1000)
        elif model_type == 'gbc':
            clf = gbc(random_state=555, verbose=True, n_estimators=int(round(x_train.shape[0]/20, 0)))
        elif model_type == 'gbr':
            clf = gbr(random_state=555, verbose=True, n_estimators=int(round(x_train.shape[0] / 20, 0)))
            use_sparse=False
        elif model_type == 'abc':
            clf = abc(random_state=555, n_estimators=int(round(x_train.shape[0]/20, 0)))
        else:
            raise ValueError('Incorrect model_type given. Cannot match [%s] to a model.' % model_type)

        # print("NaN in x_train: %s" % np.isnan(x_train.data).any())
        # print("NaN in y_train: %s" % np.isnan(y_train.data).any())
        x_train = fix_np_nan(x_train)

        print("\n----- Training Predictive Model -----")



        if selection_limit < 1.0:
            scores, p_vals = sk_feat_sel.f_regression(x_train, y_train, center=False)
            for k, v in enumerate([v[0] for v in x_columns]):
                if '_isnull' not in v: # ignore the isnull values
                    if p_vals[k] > selection_limit or p_vals[k] == np.nan:
                        x_fields.pop(v[:-2] if v[-2:] == '_t' else v)

            x_columns, x_train, x_mappings = get_vectors(df[mask_train], x_fields, x_mappings)
            x_train = fix_np_nan(x_train)

        if isinstance(clf, ridge):
            # load the diabetes datasets
            # prepare a range of alpha values to test
            alphas = np.array([100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0])
            # create and fit a ridge regression model, testing each alpha
            grid = GridSearchCV(estimator=clf, param_grid=dict(alpha=alphas))
            grid.fit(x_train, y_train)
            print(grid)
            # summarize the results of the grid search
            print('Ridge Regression Best Score:', grid.best_score_)
            print('Ridge Regression Best Alpha:', grid.best_estimator_.alpha)
            clf.alpha = grid.best_estimator_.alpha

            x_columns, x_train, x_mappings = get_vectors(df[mask_train], x_fields, x_mappings)

            x_train = fix_np_nan(x_train)

        clf.fit(x_train, y_train)

        print("------ Model Training Complete ------\n")

        try:
            with open(predictive_model_file, 'wb') as pickle_file:
                pickle.dump((clf,), pickle_file)
        except AttributeError:
            pass

    if show_model_tests:
        x_columns, x_test, x_mappings = get_vectors(df[mask_test], x_fields, x_mappings)
        y_test, y_mappings = get_vectors(df[mask_test], y_field, y_mappings, is_y=True)
        x_test = fix_np_nan(x_test)

        print("Total Named Columns: ", len(x_columns))

        coef_list = [['name'] + [v.strip() for v, b in x_columns]]

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

            print_df = pd.DataFrame(list(zip([v1 for v1, v2 in x_columns]
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
                use_sparse = True
                preds = clf.predict(x_test)
        print('R2 SCORE\n', r2_score(y_test, preds))

        if hasattr(clf, 'classes_'):
            print('CLASSES')
            print(clf.classes_)
            print('ACCURACY:\n', metrics.accuracy_score(y_test, preds))
            prediction_probs = clf.predict_proba(x_test)
            try:
                if prediction_probs.shape[1] <= 2:
                    print('ROC-CURVE AOC', metrics.roc_auc_score(y_test, prediction_probs[:, 1]))
            except ValueError as e:
                print(e)
                pass
            # if not isinstance(preds, list):
            #     preds = [preds]
            # lol = list(enumerate(preds))
            # maps = list(y_mappings.values())

            # preds = [list(y_mappings.values())[index][values] for index, values in enumerate(preds)]

            print("PREDICTION PROBABILITIES")
            print(prediction_probs)

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
                use_sparse = True
                preds = clf.predict(x_validate)
                pass
        except DeprecationWarning as e:
            print("YOU NEED TO FIX THE BELOW ERROR SINCE IT WILL BE DEPRECATED")
            print(e)
            preds = clf.predict(x_validate.reshape(-1, 1))
            pass
        y_validate.append(preds)


    if hasattr(clf, 'classes_'):
        prediction_probs = list()
        for s in range(0, int(math.ceil(len(df_validate.index)/max_slice_size))):
            prediction_probs.append(clf.predict_proba(x_validate))
        prediction_probs = np.vstack(prediction_probs)
        print('ORIGINAL PROBABILITIES')
        print(*prediction_probs[:10].round(4), sep='\n')
        print('ORIGINAL PROBABILITIES (NORMALIZED)')
        print(*[np.around(np.expm1(x), 4) for x in prediction_probs][:10], sep='\n')

        for m in y_mappings.values():
            df_probs = pd.DataFrame(data=prediction_probs.round(4)
                            , columns=['prob_' + f.lower() for f in list(m.classes_)[:prediction_probs.shape[1]]]
                            , index=df_validate.index)
            # df_validate.reset_index(inplace=True, drop=True)
            df_validate = df_validate.join(df_probs, how='inner')

    # WHY HSTACK? BECAUSE WHEN THE ndarray is 1-dimensional, apparently vstack doesn't work. FUCKING DUMB.
    y_validate = np.hstack(y_validate)

    for k, v in y_field.items():
        final_preds = y_mappings[k].inverse_transform(y_validate)
        print('FINAL PREDICTIONS')
        print(*final_preds[:10])
        df_validate.loc[:, 'pred_' + k] = final_preds

    if predict_all:
        df = df_validate
    else:
        df = df.loc[mask_validate == 0, :].append(df_validate, ignore_index=True)

    return df

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
                 , field_names: OrderedDict
                 ):
    trained_models = OrderedDict()
    for f, t in list(field_names.items()):
        if verbose:
            print("VECTORIZING: ", f)

        if isinstance(t, (list, np.ndarray)):
            vectorizer = sk_prep.LabelEncoder()
            vectorizer.fit(df[f].values.astype(str).transpose())

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
            else:
                vectorizer = sk_prep.StandardScaler()
            # vectorizer = sk_prep.MinMaxScaler()
            vectorizer.fit(df[f].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64).reshape(-1, 1))
        else:
            raise ValueError('Invalid column type provided. Choose between: \'num\', \'cat\', and \'doc\'.')
        trained_models[f] = vectorizer

    return trained_models


def get_vectors(df: pd.DataFrame
                , field_names: OrderedDict
                , trained_models: OrderedDict
                , is_y: bool=False
                ):
    final_matrix = None
    column_names = list()
    for f, t in list(field_names.items()):
        if verbose:
            print("TRANSFORMING: ", f, end='')
        isnull_series = df[f].isnull().values.astype(np.float64).reshape(-1, 1)
        if use_sparse:
            null_matrix = sparse.csc_matrix(isnull_series, dtype=np.float64)
        else:
            null_matrix = isnull_series.as_matrix().astype(np.float64)

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
                column_names.append((v, True))

        elif t == 'doc':
            matrix = trained_models[f].transform(df[f].values.astype(str)).transpose()
            # for i in range(1, matrix.shape[0]):
            #     column_names.append((f, False))
            for v in trained_models[f].get_feature_names():
                # Change to false if you want to eliminate the (MANY) possible words
                column_names.append((v
                                     .encode('ascii', errors='ignore')
                                     .decode('utf-8', errors='ignore'), True))

            if use_sparse:
                matrix = sparse.csc_matrix(matrix)

            if isnull_series.any():
                column_names.append((f + '_isnull', True))
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

                # matrix.data = np.minimum(1, np.maximum(0, np.nan_to_num(matrix.data)))
                column_names.append((f + '_t', True))

                if isnull_series.any():
                    matrix = matrix_vstack((matrix, null_matrix))
                    column_names.append((f + '_isnull', True))

        if verbose:
            print(' || Shape: {0}'.format(matrix.shape))
        if final_matrix is not None:
            final_matrix = matrix_vstack((final_matrix, matrix))
        else:
            final_matrix = matrix
    else:
        if is_y:
            return final_matrix, trained_models
        else:
            if use_sparse:
                final_matrix = sparse.csr_matrix(final_matrix)
            return column_names, final_matrix.transpose(), trained_models

def matrix_vstack(m):
    if use_sparse:
        m = sparse.vstack(m)
    else:
        m = np.vstack(m)
    return m

def reverse_enumerate(l):
   for index in reversed(range(len(l))):
      yield index, l[index]