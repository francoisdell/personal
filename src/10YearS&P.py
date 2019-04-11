__author__ = 'Andrew Mackenzie'
__author_handle__ = '@andymackenz'
__report_name__ = 'Forward S&P Price Predictions'

import matplotlib
matplotlib.use('TkAgg')
from Function_Toolbox import impute_if_any_nulls
from AssetPredictionFunctionToolbox import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from Model_Builder import ModelBuilder, ModelSet
from collections import OrderedDict
from dateutil.relativedelta import relativedelta
from Settings import Settings
import warnings
def warn(*args, **kwargs): pass; warnings.warn = warn  # This stops sklearn from bombarding you with warnings


# ==================================================================================================================== #
# ======================================   SETTINGS SPECIFICATION SECTION   ========================================== #
# ==================================================================================================================== #
# Specify whether the data will be loaded from pickle files or recollected fresh from the interwebz
rawdata_from_file = 'auto'  # True = from file || False = reload data from interweb || 'auto' = reload once per week

finaldata_from_file = False  # Whether to load the final data (post-transformations) from a pickle file

# ======================================================================================================================
# Specify the target you want to predict and the models you want to use to predict it
#
# Possible numerical models: linreg, knn_r, svr, gbr, etree_r, elastic_net, elastic_net_stacking, rfor_r, nu_svr, sgd_r,
#                            pass_agg_r, lasso_r, ridge_r, gauss_proc_r. xgb_r, neural_r{number layers}
#
# Possible categorical models: logit, knn_c, svc, gbc, etree_c, rfor_c, nu_svc, sgd_c, abc,
#                              pass_agg_c, ridge_c, gauss_proc_c. xgb_c, neural_c{number layers}
# ======================================================================================================================
do_predict_returns = False
returns_models = ModelSet(initial_models=['xgb_r', 'linreg', 'knn_r', 'neural_r2', 'etree_r'],
                          final_models=['xgb_r', 'linreg', 'knn_r', 'neural_r2'])

# returns_models = ModelSet(final_models=['rnn_r'])

do_predict_recession_flag = False
recession_flag_models = ModelSet(initial_models=[], final_models=['xgb_c', 'neural_c', 'logit', 'svc', 'knn_c'])

do_predict_recession_quarters_method = 'STACKING'  # None/False, 'SARIMAX', or 'STACKING'
recession_quarters_models = ModelSet(initial_models=['linreg', 'etree_r'],
                                     final_models=['xgb_r', 'knn_r'])

# If you didn't specify a set of models for your target above, you can specify the default models you want to use here
use_fast_models = True
use_test_models = True
use_neural_nets = True

# How many quarters to predict forward for the returns and recession flag models
returns_predict_quarters_forward = [28]
recession_flag_predict_quarters_forward = [6]

returns_comparison_asset_name = 'tsy_7yr_yield'

interaction_type = None  # Specify whether to derive pairwise interaction variables. Options: all, level_1, None
correlation_type = None  # Specify whether to derive pairwise EWM-correlation variables. Options: level_1, None
transform_vars = False  # For each x-variable, creates an additional x^2 and x^0.5 version too
trim_stdevs = 4  # Trims outlier variables according to a certain number of standard deviations (0 for skip)
calc_trend_values = False  # Add x-variables for time since the last rise, and time since the last fall
calc_std_values = True  # For each x-variable, adds an additional var for the # of stds the variable is from ewma
calc_diff_from_trend_values = True
diff_quarters = [6, 12, 18]

# If you want to remove vars that don't meet a certain significance level, set this < 1. Requiring 95% = 5.0e-2.
mb_train_pct = 0.8  # Defines the train/test split
mb_selection_limit = 0.1
mb_correlation_max = 0.90  # 1 - ((1 - 0.9) * (1 - mb_train_pct))

# VARIANCE REDUCTION. Either PCA Variance or Correlation Limits. This is the amount of explained variance you want.
correlation_method = 'corr'  # Use either None, 'corr' for correlations, or 'pca' for PCA
max_correlation = 0.90  # Must be decimal 0-1 (0 for default, which is 0.99 for PCA and 0.80 for corr)

# DIMENSION REDUCTION. Either PCA Variance or Correlation Rankings.
dimension_method = 'corr'  # Use either None, 'corr' for correlations, or 'pca' for PCA
max_variables = 0.5  # Positive val for max # of variables you want (0 = n_observations^0.5) (0<x<1 = n_observations^x)

# Variables specifying what kinds of predictions to run, and for what time period
real = False
sp_field_name = 'sp500'
recession_field_name = 'recession_usa'
prev_rec_field_name = recession_field_name + '_time_since_prev'
next_rec_field_name = recession_field_name + '_time_until_next'
verbose = False
ewm_alpha = 0.06  # Halflife for EWM calculations
use_vwma = True
use_ivwma = True
use_ewma = False
default_imputer = 'fancyimpute'  # 'fancyimpute' or 'knnimpute'. knnimpute is generally much faster, if less ideal.
stack_include_preds = False
final_include_data = True
show_plots = False
exec_time = dt.strftime(dt.now(), '%Y-%m-%d-%H-%M-%S')


# ==================================================================================================================== #
# ==================================================================================================================== #
# ==================================================================================================================== #


def predict_returns(df: pd.DataFrame,
                    x_names: list,
                    y_field_name: str,
                    model_set: ModelSet,
                    quarters_forward: int,
                    show_plot: bool = False,
                    comparison_asset_name: str='tsy_10yr_yield') \
        -> (OrderedDict, str):
    print("\n-----------------------"
          "\n   QUARTERS FORWARD: {0} "
          "\n-----------------------".format(quarters_forward))

    if quarters_forward > 0:
        forward_y_field_name = '{0}_{1}qtr'.format(y_field_name, quarters_forward)
        df[forward_y_field_name] = 1 - pd.Series(df[y_field_name].shift(quarters_forward) / df[y_field_name]).pow(
            1 / float(quarters_forward)).shift(-quarters_forward)

    else:
        forward_y_field_name = y_field_name
        df[forward_y_field_name] = df[y_field_name]

    for v in x_names:
        if not np.isfinite(df[v].astype(float)).all() or not np.isfinite(df[v].astype(float).sum()):
            print('Found Series with non-finite values:{0}'.format(v))
            print(*df[v].values, sep='\n')
            raise Exception("Can't proceed until you fix the Series.")

    train_mask = ~df[forward_y_field_name].isnull()
    #################################
    # USING THE MODEL BUILDER CLASS #
    #################################
    y_field = OrderedDict([(forward_y_field_name, 'num')])
    x_fields = OrderedDict([(v, 'num') for v in x_names])

    report_name = 'returns_{0}qtr_{1}'.format(quarters_forward, model_set)
    m = ModelBuilder(df,
                     x_fields=x_fields,
                     y_field=y_field,
                     model_type=model_set,
                     report_name=report_name,
                     show_model_tests=True,
                     retrain_model=True,
                     selection_limit=mb_selection_limit,
                     predict_all=True,
                     verbose=verbose,
                     train_pct=mb_train_pct,
                     random_train_test=False,
                     stack_include_preds=stack_include_preds,
                     final_include_data=final_include_data,
                     correlation_max=mb_correlation_max,
                     # correlation_method='pca',
                     use_test_set=True,
                     use_sparse=False,
                     codify_nulls=False
                     )

    for final_model_name in m.predict():

        pred_df = m.df.copy()
        forward_y_field_name_pred = 'pred_' + forward_y_field_name
        #################################

        invest_amt_per_qtr = 5000
        pred_df['invest_strat_cash'] = invest_amt_per_qtr
        pred_df['invest_strat_basic'] = invest_amt_per_qtr
        pred_df['invest_strat_mixed'] = invest_amt_per_qtr
        pred_df['invest_strat_equity'] = invest_amt_per_qtr
        pred_df['invest_strat_tsy'] = invest_amt_per_qtr

        mean_y_field_return = pred_df[forward_y_field_name].mean()

        for idx in range(pred_df.shape[0] - quarters_forward):
            comparison_yield = pred_df.ix[idx, comparison_asset_name]
            comparison_return = 1 + comparison_yield
            forward_y_return = 1 + pred_df.ix[idx, forward_y_field_name]
            if pd.isna(comparison_return) or pd.isna(forward_y_return):
                continue
            else:
                future_idx = idx + quarters_forward

                pred_df.ix[future_idx, 'invest_strat_cash'] += pred_df.ix[idx, 'invest_strat_cash']

                # Implement a basic investment strategy where if the tsy yield falls below the average historical
                # of the target asset, invest in the asset. Otherwise invest in treasuries.
                if mean_y_field_return >= comparison_yield:
                    pred_df.ix[future_idx, 'invest_strat_basic'] += \
                        pred_df.ix[idx, 'invest_strat_basic'] * (forward_y_return ** quarters_forward)
                else:
                    pred_df.ix[future_idx, 'invest_strat_basic'] += \
                        pred_df.ix[idx, 'invest_strat_basic'] * (comparison_return ** quarters_forward)

                # Implement a mixed investment strategy where if the tsy yield falls below the model's predicted
                # forward return of the target asset, invest in the asset. Otherwise invest in treasuries.
                if pred_df.ix[idx, forward_y_field_name_pred] >= comparison_yield:
                    pred_df.ix[future_idx, 'invest_strat_mixed'] += \
                        pred_df.ix[idx, 'invest_strat_mixed'] * (forward_y_return ** quarters_forward)
                else:
                    pred_df.ix[future_idx, 'invest_strat_mixed'] += \
                        pred_df.ix[idx, 'invest_strat_mixed'] * (comparison_return ** quarters_forward)

                # Implement a strategy of investing only in the target asset
                pred_df.ix[future_idx, 'invest_strat_equity'] += \
                    pred_df.ix[idx, 'invest_strat_equity'] * (forward_y_return ** quarters_forward)

                # Implement a strategy of investing only in the 10-year treasury
                pred_df.ix[future_idx, 'invest_strat_tsy'] += \
                    pred_df.ix[idx, 'invest_strat_tsy'] * (comparison_return ** quarters_forward)

        pred_df['total_strat_cash'] = pred_df['invest_strat_cash'][train_mask].rolling(quarters_forward).sum()
        pred_df['total_strat_basic'] = pred_df['invest_strat_basic'][train_mask].rolling(quarters_forward).sum()
        pred_df['total_strat_mixed'] = pred_df['invest_strat_mixed'][train_mask].rolling(quarters_forward).sum()
        pred_df['total_strat_equity'] = pred_df['invest_strat_equity'][train_mask].rolling(quarters_forward).sum()
        pred_df['total_strat_tsy'] = pred_df['invest_strat_tsy'][train_mask].rolling(quarters_forward).sum()

        final_return_strat_cash = pred_df['total_strat_cash'][train_mask].iloc[-1]
        final_return_strat_basic = pred_df['total_strat_basic'][train_mask].iloc[-1]
        final_return_strat_mixed = pred_df['total_strat_mixed'][train_mask].iloc[-1]
        final_return_strat_equity = pred_df['total_strat_equity'][train_mask].iloc[-1]
        final_return_strat_tsy = pred_df['total_strat_tsy'][train_mask].iloc[-1]

        print('\nFinal Results')
        print('Cash Strategy: %.2fM' % (final_return_strat_cash / 1000000))
        print('Basic Strategy: %.2fM' % (final_return_strat_basic / 1000000))
        print('Mixed Strategy: %.2fM' % (final_return_strat_mixed / 1000000))
        print('%s: %.2fM' % (y_field_name, final_return_strat_equity / 1000000))
        print('10Yr Tsy: %.2fM' % (final_return_strat_tsy / 1000000))

        print('\nModel Comparisons')
        print('Mixed vs. Cash Strat: %.2f' % (final_return_strat_mixed / final_return_strat_cash))
        print('Mixed vs. Basic Strat: %.2f' % (final_return_strat_mixed / final_return_strat_basic))
        print('Mixed vs. Equity Strat: %.2f' % (final_return_strat_mixed / final_return_strat_equity))
        print('Mixed vs. Treasury Strat: %.2f' % (final_return_strat_mixed / final_return_strat_tsy))

        forward_y_field_name_pred_fut = forward_y_field_name_pred + '_fut'
        pred_df[forward_y_field_name_pred_fut] = pred_df[forward_y_field_name_pred].astype(np.float16)
        pred_df[forward_y_field_name_pred_fut][train_mask] = np.nan

        print(max(pred_df.index))
        pred_df = pred_df.reindex(
            pd.DatetimeIndex(start=pred_df.index.min(),
                             end=max(pred_df.index) + relativedelta(months=quarters_forward * 3),
                             freq='1Q',
                             dtype=dt.date))

        y_field_name_pred = '{0}_pred'.format(y_field_name)
        pred_df[y_field_name_pred] = (pred_df[y_field_name] * pred_df[forward_y_field_name_pred]
                                      .add(1).pow(quarters_forward)).shift(quarters_forward)

        report_name = 'returns_{0}qtr_{1}_{2}'.format(quarters_forward, model_set, final_model_name)

        chart_file_name = '_'.join([exec_time, forward_y_field_name_pred, report_name, str(quarters_forward)])
        # if quarters_forward in [10]:
        chart(pred_df,
              ys=[[forward_y_field_name, forward_y_field_name_pred, 'tsy_10yr_yield'],
                    [y_field_name, y_field_name_pred]],
              invert=[False, False],
              log_scale=[False, True],
              save_addr=os.path.join(s.get_reports_dir(), chart_file_name),
              title='_'.join([y_field_name, report_name]),
              show=show_plot)

        yield (OrderedDict(((forward_y_field_name, pred_df[forward_y_field_name]),
                            (forward_y_field_name_pred, pred_df[forward_y_field_name_pred]),
                            (y_field_name_pred, pred_df[y_field_name_pred]))),
               report_name)


def predict_recession(df: pd.DataFrame,
                      x_names: list,
                      y_field_name: str,
                      quarters_forward: int,
                      model_set: ModelSet,
                      show_plot: bool = False) \
        -> (OrderedDict, str):
    print("\n-----------------------"
          "\n   QUARTERS FORWARD: {0} "
          "\n-----------------------".format(quarters_forward))

    if quarters_forward > 0:
        forward_y_field_name = '{0}_{1}qtr'.format(y_field_name, quarters_forward)
        df[forward_y_field_name] = df[y_field_name].shift(-quarters_forward)

    else:
        forward_y_field_name = y_field_name
        df[forward_y_field_name] = df[y_field_name]

    for v in x_names:
        if not np.isfinite(df[v]).all() or not np.isfinite(df[v].sum()):
            print('Found Series with non-finite values:{0}'.format(v))
            print(*df[v].values, sep='\n')
            raise Exception("Can't proceed until you fix the Series.")

    #################################
    # USING THE MODEL BUILDER CLASS #
    #################################
    y_field = OrderedDict([(forward_y_field_name, 'cat')])
    x_fields = OrderedDict([(v, 'num') for v in x_names])

    report_name = 'recession_{0}qtr_{1}'.format(quarters_forward, model_set)

    m = ModelBuilder(df,
                     x_fields=x_fields,
                     y_field=y_field,
                     model_type=model_set,
                     report_name=report_name,
                     show_model_tests=True,
                     retrain_model=False,
                     selection_limit=mb_selection_limit,
                     predict_all=True,
                     verbose=verbose,
                     train_pct=mb_train_pct,
                     random_train_test=False,
                     pca_explained_var=1.0,
                     stack_include_preds=stack_include_preds,
                     final_include_data=final_include_data,
                     use_test_set=True,
                     correlation_method='corr',
                     correlation_max=mb_correlation_max,
                     use_sparse=False,
                     codify_nulls=False
                     # cross_val_iters=(20,20),
                     )

    for final_model_name in m.predict():

        forward_y_field_name_pred = 'pred_' + forward_y_field_name
        #################################

        print(f'Latest date of prediction dataframe: {max(m.df.index)}')
        pred_df = m.df.reindex(
            pd.DatetimeIndex(start=m.df.index.min(),
                             end=max(m.df.index) + relativedelta(months=quarters_forward * 3),
                             freq='1Q',
                             dtype=dt.date))

        y_field_name_pred = '{0}_pred'.format(y_field_name)
        pred_df[y_field_name_pred] = pred_df[forward_y_field_name_pred].shift(quarters_forward)
        chart_y_field_names = [y_field_name, y_field_name_pred]

        y_pred_names = m.new_fields
        y_prob_field_name = '{0}_prob_1.0'.format(forward_y_field_name)
        if y_prob_field_name in y_pred_names:
            pred_df[y_prob_field_name] = pred_df[y_prob_field_name].shift(quarters_forward)
            chart_y_field_names += [y_prob_field_name]

        report_name = 'recession_{0}qtr_{1}_{2}'.format(quarters_forward, model_set, final_model_name)

        # if quarters_forward in [10]:
        chart_file_name = '_'.join([exec_time, forward_y_field_name_pred, report_name])

        chart(df,
              ys=[['sp500'], chart_y_field_names],
              invert=[False, False],
              log_scale=[True, False],
              save_addr=os.path.join(s.get_reports_dir(), chart_file_name),
              title='_'.join([y_field_name, report_name]),
              show=show_plot)

        yield (OrderedDict((
            (forward_y_field_name, pred_df[forward_y_field_name].astype(float)),
            (forward_y_field_name_pred, pred_df[forward_y_field_name_pred].astype(float)),
            (y_field_name_pred, pred_df[y_field_name_pred].astype(float)),
            (y_prob_field_name, pred_df[y_prob_field_name].astype(float)),
        )),
               report_name)


def predict_recession_time(df: pd.DataFrame,
                           x_names: list,
                           y_field_name: str,
                           model_set: ModelSet,
                           show_plot: bool = False) \
        -> (OrderedDict, str):
    train_mask = ~df[y_field_name].isnull()

    for v in x_names:
        if not np.isfinite(df[v]).all() or not np.isfinite(df[v].sum()):
            print('Found Series with non-finite values:{0}'.format(v))
            print(*df[v].values, sep='\n')
            raise Exception("Can't proceed until you fix the Series.")

    #################################
    # USING THE MODEL BUILDER CLASS #
    #################################
    y_field = OrderedDict([(y_field_name, 'num')])
    x_fields = OrderedDict([(v, 'num') for v in x_names])

    if do_predict_recession_quarters_method == 'SARIMAX':
        # Generate all different combinations of p, q and q triplets
        # Generate all different combinations of seasonal p, q and q triplets

        p = d = q = range(1, 2)  # Change to 0, 2
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]

        best_model = None
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(endog=df.loc[train_mask, y_field_name],
                                  exog=df.loc[train_mask, x_names],
                                  order=param,
                                  seasonal_order=param_seasonal,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)

                    results = mod.fit()

                    if best_model is None or results.aic < best_model.aic:
                        best_model = results

                    print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                except Exception as e:
                    continue

        start = df.loc[train_mask].index[-1]
        end = df.loc[~train_mask].index[-1]
        start_pred = pd.DatetimeIndex(df.loc[train_mask].index.shift(1, 'Q'),
                                      dtype=dt.date)[-1]
        sarimax_preds = best_model.get_prediction(start=start,
                                                  end=end,
                                                  exog=df.loc[start_pred:end, x_names],
                                                  dynamic=True)
        print(sarimax_preds)

        y_field_name_pred = 'pred_' + y_field_name

        df.loc[:, y_field_name_pred] = df.loc[:, y_field_name]

        df.loc[sarimax_preds.predicted_mean.index.values, y_field_name_pred] = sarimax_preds.predicted_mean.values
        report_name = 'time_to_next_recession_sarimax'

        # if quarters_forward in [10]:
        chart_file_name = '_'.join([exec_time, y_field_name_pred, report_name])
        chart(df,
              ys=[['sp500'], [y_field_name, y_field_name_pred, 'tsy_10yr_yield']],
              invert=[False, False],
              log_scale=[True, False],
              save_addr=os.path.join(s.get_reports_dir(), chart_file_name),
              title='_'.join([y_field_name, report_name]),
              show=show_plot)

        yield (OrderedDict((
            (y_field_name, df[y_field_name].astype(float)),
            (y_field_name_pred, df[y_field_name_pred].astype(float)),
        )),
               report_name)

    elif do_predict_recession_quarters_method == 'STACKING':

        forward_y_field_name = y_field_name
        df[forward_y_field_name] = df[y_field_name]

        for v in x_names:
            if not np.isfinite(df[v]).all() or not np.isfinite(df[v].sum()):
                print('Found Series with non-finite values:{0}'.format(v))
                print(*df[v].values, sep='\n')
                raise Exception("Can't proceed until you fix the Series.")

        #################################
        # USING THE MODEL BUILDER CLASS #
        #################################
        y_field = OrderedDict([(forward_y_field_name, 'num')])
        x_fields = OrderedDict([(v, 'num') for v in x_names])

        report_name = 'next_recession_qtrs_{0}'.format(model_set)
        m = ModelBuilder(df,
                         x_fields=x_fields,
                         y_field=y_field,
                         model_type=model_set,
                         report_name=report_name,
                         show_model_tests=True,
                         retrain_model=False,
                         selection_limit=mb_selection_limit,
                         predict_all=True,
                         verbose=verbose,
                         train_pct=mb_train_pct,
                         random_train_test=False,
                         pca_explained_var=1.0,
                         stack_include_preds=stack_include_preds,
                         final_include_data=final_include_data,
                         use_test_set=True,
                         correlation_method='corr',
                         correlation_max=mb_correlation_max,
                         use_sparse=False,
                         codify_nulls=False
                         # cross_val_iters=(20,20),
                         )
        for final_model_name in m.predict():

            forward_y_field_name_pred = 'pred_' + forward_y_field_name
            #################################
            pred_df = m.df
            print(max(pred_df.index))
            y_pred_names = m.new_fields
            y_field_name_pred = y_pred_names[0]
            chart_y_field_names = [y_field_name, y_field_name_pred]

            y_prob_field_name = '{0}_prob_1.0'.format(forward_y_field_name)
            if y_prob_field_name in y_pred_names:
                chart_y_field_names += [y_prob_field_name]

            report_name = 'next_recession_qtrs_{0}_{1}'.format(model_set, final_model_name)
            chart_file_name = '_'.join([exec_time, y_field_name_pred, report_name])
            chart(pred_df,
                  ys=[['sp500'], [y_field_name, y_field_name_pred, 'tsy_10yr_yield']],
                  invert=[False, False],
                  log_scale=[True, False],
                  save_addr=os.path.join(s.get_reports_dir(), chart_file_name),
                  title='_'.join([y_field_name, report_name]),
                  show=show_plot)

            yield (OrderedDict((
                (y_field_name, df[y_field_name].astype(float)),
                (y_field_name_pred, df[y_field_name_pred].astype(float)),
            )),
                   report_name)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# START THE STUFF
# START THE STUFF
# START THE STUFF
########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':

    s = Settings(report_name='10YearS&P')

    # SET THE MODELS FOR THE RETURNS PREDICTION METHOD
    if do_predict_returns:
        if not returns_models:
            if use_test_models:
                initial_models = []
                final_models = ['rnn_r']
            elif use_fast_models:
                initial_models = ['linreg']
                final_models = ['xgb_r']
                final_include_data = True
            else:
                initial_models = ['rfor_r', 'gbr', 'linreg', 'elastic_net', 'knn_r', 'svr']
                if (correlation_method is not None and max_correlation < 1.) or max_variables == 0:
                    # initial_models.extend(['gauss_proc_r'])
                    if use_neural_nets:
                        initial_models.extend(['neural_r_2'])

                final_models = ['linreg', 'svr', 'xgb_r']
                if (not final_include_data) or (
                        correlation_method is not None and max_correlation < 1.) or max_variables == 0:
                    final_models.extend(['gauss_proc_r'])
                    if use_neural_nets:
                        final_models.extend(['neural_r'])

            returns_models = ModelSet(final_models=final_models, initial_models=initial_models)

    # SET THE MODELS FOR THE RECESSION PREDICTION METHOD
    if do_predict_recession_flag:
        if not recession_flag_models:
            if use_fast_models:
                initial_models = ['logit']
                final_models = ['xgb_c']
                final_include_data = True
            else:
                initial_models = ['logit', 'etree_c', 'nearest_centroid', 'gbc', 'bernoulli_nb', 'svc',
                                  'rfor_c']  # pass_agg_c
                # initial_models=['logit','etree_c']
                if (correlation_method is not None and max_correlation < 1.) or max_variables == 0:
                    initial_models.extend(['gauss_proc_c'])
                    if use_neural_nets:
                        initial_models.extend(['neural_c_2'])

                # BEST MODELS: logit, svc, sgd_c, neural_c, gauss_proc_c
                # Overall best model: svc???
                final_models = ['logit', 'svc', 'xgb_c']
                if (not final_include_data) or max_correlation < 1. or max_variables == 0:
                    final_models.extend(['gauss_proc_c'])
                    if use_neural_nets:
                        final_models.extend(['neural_c'])

            # initial_models=['logit','etree_c','nearest_centroid','gbc','bernoulli_nb','svc','pass_agg_c','gauss_proc_c']
            recession_flag_models = ModelSet(final_models=final_models, initial_models=initial_models)

    # SET THE MODELS FOR THE NEXT-RECESSION PREDICTION METHOD
    if do_predict_recession_quarters_method:
        if not recession_quarters_models:
            if use_fast_models:
                initial_models = ['rfor_r']
                final_models = ['xgb_r']  # ['linreg']
                final_include_data = True
            else:
                initial_models = ['rfor_r', 'knn_r', 'elastic_net', 'xgb_r']
                if (correlation_method is not None and max_correlation < 1.) or max_variables == 0:
                    initial_models.extend(['gauss_proc_r'])
                    if use_neural_nets:
                        initial_models.extend(['neural_r_2'])

                final_models = ['linreg', 'elastic_net_stacking', 'xgb_r']
                if (correlation_method is not None and max_correlation < 1.) or max_variables == 0:
                    final_models.extend(['gauss_proc_r'])
                    if use_neural_nets:
                        final_models.extend(['neural_r'])

            recession_quarters_models = ModelSet(final_models=final_models, initial_models=initial_models)

    try:
        run_hist = pickle_load('run_hist.p')
    except FileNotFoundError:
        run_hist = dict()

    if rawdata_from_file == 'auto':
        if 'rawdata_last_load' in run_hist and run_hist['rawdata_last_load'] > dt.now() - td(days=7):
            rawdata_from_file = True
        else:
            rawdata_from_file = False

    if real:
        sp_field_name += '_real'

    # use the list to filter the local namespace
    # safe_func_list = ['first', 'last', 'mean']
    # safe_funcs = dict([(k, locals().get(k, None)) for k in safe_func_list])

    data_sources = dict()
    data_sources['cpi_urb_nonvol'] = DataSource('CPILFESL', 'fred')
    data_sources['netexp_nom'] = DataSource('NETEXP', 'fred')
    data_sources['gdp_nom'] = DataSource('GDP', 'fred')
    data_sources['sp500'] = DataSource('P', 'schiller')
    data_sources['cape'] = DataSource('CAPE', 'schiller')
    data_sources['cape_tr'] = DataSource('CAPE.1', 'schiller')
    data_sources['tsy_3mo_yield'] = DataSource('DGS3MO', 'fred')
    data_sources['tsy_6mo_yield'] = DataSource('DGS6MO', 'fred')
    data_sources['tsy_1yr_yield'] = DataSource('DGS1', 'fred')
    data_sources['tsy_2yr_yield'] = DataSource('DGS2', 'fred')
    data_sources['tsy_3yr_yield'] = DataSource('DGS3', 'fred')
    data_sources['tsy_5yr_yield'] = DataSource('DGS5', 'fred')
    data_sources['tsy_7yr_yield'] = DataSource('DGS7', 'fred')
    data_sources['tsy_10yr_yield'] = DataSource('DGS10', 'fred')
    data_sources['tsy_30yr_yield'] = DataSource('DGS30', 'fred')
    data_sources['fed_funds_rate'] = DataSource('FEDFUNDS', 'fred')
    data_sources['gold_fix_3pm'] = DataSource('GOLDPMGBD228NLBM', 'fred')
    data_sources['unempl_rate'] = DataSource('UNRATE', 'fred')
    data_sources['industrial_prod'] = DataSource('INDPRO', 'fred')
    data_sources['empl_construction'] = DataSource('USCONS', 'fred')
    data_sources['equity_alloc'] = DataSource(None, calc_equity_alloc)
    data_sources['fed_reserves_tot'] = DataSource('RESBALNS', 'fred')
    data_sources['monetary_base_tot'] = DataSource('BOGMBASE', 'fred')
    data_sources['monetary_base_balances'] = DataSource('BOGMBBMW', 'fred')
    data_sources['fed_excess_reserves'] = DataSource('EXCSRESNS', 'fred')
    data_sources['sp500_peratio'] = DataSource('MULTPL/SHILLER_PE_RATIO_MONTH', 'quandl')
    data_sources['housing_starts'] = DataSource('HOUST', 'fred')
    data_sources['housing_supply'] = DataSource('MSACSR', 'fred')
    data_sources['real_estate_loans'] = DataSource('REALLN', 'fred')
    data_sources['med_house_price'] = DataSource('MSPNHSUS', 'fred')
    data_sources['med_family_income'] = DataSource('MEFAINUSA646N', 'fred')
    data_sources['combanks_business_loans'] = DataSource('BUSLOANS', 'fred')
    data_sources['combanks_assets_tot'] = DataSource('TLAACBW027SBOG', 'fred')
    data_sources['combanks_realestate_loans'] = DataSource('RELACBQ158SBOG', 'fred')
    data_sources['mortage_debt_individuals'] = DataSource('MDOTHIOH', 'fred')
    data_sources['capacity_util_tot'] = DataSource('CAPUTLB50001SQ', 'fred')
    data_sources['capacity_util_mfg'] = DataSource('CUMFNS', 'fred')
    data_sources['capacity_util_chem'] = DataSource('CAPUTLG325S', 'fred')
    data_sources['foreign_dir_invest'] = DataSource('ROWFDNQ027S', 'fred')
    data_sources['pers_savings_rt'] = DataSource('PSAVERT', 'fred')
    data_sources['gross_savings'] = DataSource('GSAVE', 'fred')
    data_sources['profits_corp_pretax'] = DataSource('A446RC1Q027SBEA', 'fred')
    # data_sources['tax_receipts_tot'] = DataSource('W006RC1Q027SBEA', 'fred')
    data_sources['nonfin_equity'] = DataSource('MVEONWMVBSNNCB', 'fred')
    data_sources['nonfin_networth'] = DataSource('TNWMVBSNNCB', 'fred')
    data_sources['nonfin_pretax_profit'] = DataSource('NFCPATAX', 'fred')
    data_sources['mzm_velocity'] = DataSource('MZMV', 'fred')
    data_sources['m2_velocity'] = DataSource('M2V', 'fred')
    data_sources['m1_velocity'] = DataSource('M1V', 'fred')
    data_sources['mzm_moneystock'] = DataSource('MZMSL', 'fred')
    data_sources['m2_moneystock'] = DataSource('M2NS', 'fred')
    data_sources['m1_moneystock'] = DataSource('M1NS', 'fred')
    # data_sources['wrk_age_pop_pct'] = DataSource('SP.POP.1564.TO.ZS', 'worldbank', rerun=True)
    data_sources['wrk_age_pop'] = DataSource('LFWA64TTUSM647N', 'fred')
    data_sources['employment_pop_ratio'] = DataSource('EMRATIO', 'fred')
    # data_sources['nyse_margin_debt'] = DataSource('Margin debt', get_nyse_margin_debt)
    # data_sources['nyse_margin_credit'] = DataSource('Credit balances in margin accounts', get_nyse_margin_debt)
    data_sources['us_dollar_index'] = DataSource('DTWEXM', 'fred')
    data_sources[recession_field_name] = DataSource('USREC', 'fred')
    data_sources['nonfin_gross_val'] = DataSource('A455RC1Q027SBEA', 'fred')
    data_sources['nonfin_equity_liability'] = DataSource('NCBEILQ027S', 'fred')
    data_sources['fin_equity_liability'] = DataSource('FBCELLQ027S', 'fred')
    data_sources['total_market_cap_usa'] = DataSource('WILL5000INDFC', 'fred')

    ds_names = [k for k in data_sources.keys()]

    raw_data_file = 'sp500_hist.p'
    resave_data = False
    try:
        if rawdata_from_file:
            (data_sources_temp,) = pickle_load(raw_data_file)
            print('Per settings, loaded data from file [{0}]'.format(rawdata_from_file))
            for k, v in data_sources.items():
                if k not in data_sources_temp:
                    print('New data source added: [{0}]'.format(k))
                    data_sources_temp[k] = data_sources[k]
                elif v.rerun is True:
                    print('New data source added: [{0}]'.format(k))
                    data_sources_temp[k] = data_sources[k]
            data_sources = data_sources_temp
        else:
            raise ValueError('Per Settings, Reloading Data From Yahoo Finance/FRED/Everything Else.')
    except Exception as e:
        print(e)

    for k, ds in data_sources.items():
        if len(ds.data) == 0:  # check if data is empty (e.g. if index has a length of 0)
            ds.collect_data()
            resave_data = True
        data_sources[k] = ds

    if resave_data:
        pickle_dump((data_sources,), raw_data_file)
        run_hist['rawdata_last_load'] = dt.now()
        pickle_dump(run_hist, 'run_hist.p')
        # modified_z_score = 0.6745 * abs_dev / y_mad
        # modified_z_score[y == m] = 0
        # return modified_z_score > thresh


    # from pandas_datareader import data, wb
    # lol = data.DataReader(['TSLA'], 'yahoo', start_dt, end_dt)

    final_data_file = 'sp500_final_data.p'
    try:
        if finaldata_from_file:
            df, x_names = pickle_load(final_data_file)
            print('Per settings, loaded data from file [{0}]'.format(final_data_file))
        else:
            raise ValueError('Per Settings, Reloading Data From Yahoo Finance/FRED/Everything Else.')
    except Exception as e:
        print(e)
        df = pd.DataFrame()
        for k, ds in data_sources.items():
            if ds.provider in ('eod_hist', 'fred', 'schiller'):
                df[k] = make_qtrly(ds.data, 'first', name=k)
            else:
                df[k] = ds.data

        end_dt = df.index.max()
        df = df[:-1]

        df['tsy_3mo_yield'] = df['tsy_3mo_yield'] / 100.
        df['tsy_6mo_yield'] = df['tsy_6mo_yield'] / 100.
        df['tsy_1yr_yield'] = df['tsy_1yr_yield'] / 100.
        df['tsy_2yr_yield'] = df['tsy_2yr_yield'] / 100.
        df['tsy_3yr_yield'] = df['tsy_3yr_yield'] / 100.
        df['tsy_5yr_yield'] = df['tsy_5yr_yield'] / 100.
        df['tsy_7yr_yield'] = df['tsy_7yr_yield'] / 100.
        df['tsy_10yr_yield'] = df['tsy_10yr_yield'] / 100.
        df['tsy_30yr_yield'] = df['tsy_30yr_yield'] / 100.
        df['netexp_pct_of_gdp'] = (df['netexp_nom'] / df['gdp_nom'])
        df['base_minus_fed_res_tot'] = df['monetary_base_tot'].values - df['fed_reserves_tot']
        df['med_family_income_vs_house_price'] = df['med_house_price'] / df['med_family_income']
        df['real_med_family_income'] = df['med_family_income'] / df['cpi_urb_nonvol']
        df['tsy_10yr_minus_cpi'] = df['tsy_10yr_yield'] - df['cpi_urb_nonvol']
        df['tsy_10yr_minus_fed_funds_rate'] = df['tsy_10yr_yield'] - df['fed_funds_rate']
        df['tsy_2y10y_curve'] = (df['tsy_10yr_yield'] - df['tsy_2yr_yield']) / df['tsy_10yr_yield']
        df['tobin_q'] = [math.sqrt(x * y) for x, y in df.ix[:, ['nonfin_equity', 'nonfin_networth']].values]  # geo mean
        df['corp_profit_margins'] = df['nonfin_pretax_profit'] / df['gdp_nom']
        df['mzm_usage'] = df['mzm_velocity'] * df['mzm_moneystock']
        df['m1_usage'] = df['m1_velocity'] * df['m1_moneystock']
        df['m2_usage'] = df['m2_velocity'] * df['m2_moneystock']
        # df['nyse_margin_debt_ratio'] = df['nyse_margin_debt'] / df['nyse_margin_credit']
        df['corp_eq_div_nom_gdp'] = df['fin_equity_liability'] / df['gdp_nom']
        df['equities_div_gdp'] = df['sp500'] / df['gdp_nom']

        x_names = [
            'equity_alloc',
            # 'tsy_10yr_yield', # Treasury prices have been generally increasing over the time period. Don't use.,
            # 'tsy_5yr_yield', # Treasury prices have been generally increasing over the time period. Don't use.,
            # 'tsy_3mo_yield', # Treasury prices have been generally increasing over the time period. Don't use.
            'unempl_rate',
            # , 'empl_construction'  # Construction employees heave been generally increasing over the time period. Don't use.,
            'sp500_peratio',
            'capacity_util_mfg',
            'capacity_util_chem',
            # , 'gold_fix_3pm' # Gold price has been generally increasing over the time period. Don't use.
            # , 'fed_funds_rate' # Fed funds rate has been generally declining over the time period. Don't use.,
            'tsy_2y10y_curve',
            'industrial_prod',
            'tsy_10yr_minus_fed_funds_rate',
            'tsy_10yr_minus_cpi',
            # 'netexp_pct_of_gdp',  # Will cause infinite values when used with SHIFT (really any y/y compare)
            # 'gdp_nom',  # GDP is generally always rising. Don't use.
            # 'netexp_nom',  # Will cause infinite values when used with SHIFT (really any y/y compare)
            # 'base_minus_fed_res_adj',  # May also make the models go FUCKING CRAZY # Not much history
            # 'tsy_30yr_yield',  # Not much history,
            'med_family_income_vs_house_price',
            'pers_savings_rt',
            'corp_profit_margins',
            # 'cape',  # Deprecated in favor of cape_tr. See schiller's website for the details.
            'cape_tr',
            'tobin_q',
            # 'mzm_velocity',
            # 'm2_velocity',
            # 'm1_velocity',
            'mzm_usage',
            'm2_usage',
            'm1_usage',
            'employment_pop_ratio',
            'nonfin_gross_val',
            'corp_eq_div_nom_gdp',
            'equities_div_gdp',
        ]

        empty_cols = [c for c in df.columns.values if all(df[c].isnull())]
        if len(empty_cols) > 0:
            raise Exception("Empty columns in final dataframe: {0}".format(empty_cols))

        for v in x_names:
            min_dt = df[v][~df[v].isnull()].index.values[0]
            print('Earliest date for series [{0}] is: {1}'.format(v, min_dt))

        # Filter to only periods of history for which there are at least 3 data points in that period. The rest will be imputed.
        # non_null_mask = ~pd.isnull(df.loc[:, x_names]).all(axis=1).values  # Only periods where ALL data points exist
        non_null_mask = [x >= 3 for x in df.loc[:, x_names].count(axis=1).values]  # Periods where >=3 data points exist
        # non_null_mask = ~pd.isnull(df.loc[:, x_names]).any(axis=1).values  # Periods where ANY data points exist
        df = df.loc[non_null_mask, :]

        # print(df['equity_alloc'])

        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]
        # print(df['equity_alloc'])
        ##########################################################################################################
        # Derive trend metric variables
        ##########################################################################################################
        trend_x_names = []
        if calc_trend_values:
            print('Deriving trend variables.')
            # Add x-variables for time since the last rise, and time since the last fall, in the SP500
            # for x in ['sp500']:
            for x in x_names.copy():
                temp_df, new_x_names = get_trend_variables(df[x], x)

                # print(new_x_names)
                df = df.join(temp_df[new_x_names], how='inner')
                trend_x_names.extend(new_x_names)

        if calc_std_values:
            print('Deriving STD values.')

            # Add x-variables for time since the last rise, and time since the last fall, in the SP500
            # for x in ['sp500']:
            for x in x_names.copy():
                temp_df, new_x_names = get_std_from_ewma(df[x], x)

                # print(new_x_names)
                df = df.join(temp_df[new_x_names], how='inner')
                trend_x_names.extend(new_x_names)

        if calc_diff_from_trend_values:
            print('Deriving Difference-From-Mean values.')
            # Add x-variables for time since the last rise, and time since the last fall, in the SP500
            # for x in ['sp500']:
            for x in x_names.copy():
                if not df[x].isnull().any():
                    temp_df, new_x_names = get_diff_from_trend(df[x])

                    # print(new_x_names)
                    df = df.join(temp_df[new_x_names], how='inner')
                    trend_x_names.extend(new_x_names)

        ##########################################################################################################
        # Derive difference variables to demonstrate changes from period to period
        ##########################################################################################################
        print('Adding x-year diff terms.')
        diff_x_names = [
            'sp500',
            'gdp_nom',
            'equity_alloc',
            'cpi_urb_nonvol',
            'empl_construction',
            'industrial_prod',
            'housing_starts',
            'housing_supply',
            'med_house_price',
            'med_family_income',
            'unempl_rate',
            'tsy_10yr_yield',
            'tsy_5yr_yield',
            'tsy_1yr_yield',
            'tsy_10yr_minus_fed_funds_rate',
            'tsy_10yr_minus_cpi',
            'real_med_family_income',
            'combanks_business_loans',
            'combanks_assets_tot',
            'mortage_debt_individuals',
            'real_estate_loans',
            'foreign_dir_invest',
            'pers_savings_rt',
            'gross_savings',
            'profits_corp_pretax',
            'fed_funds_rate',
            'gold_fix_3pm',
            'corp_profit_margins',
            # 'cape',  # Deprecated in favor of cape_tr. See schiller's website for the details.
            'cape_tr',
            'tobin_q',
            'mzm_usage',
            'm2_usage',
            'm1_usage',
            'employment_pop_ratio',
            # 'nyse_margin_debt',
            'us_dollar_index',
            'nonfin_gross_val',
            'corp_eq_div_nom_gdp',
            'equities_div_gdp',
            # 'nyse_margin_credit',  # has an odd discontunuity in the credit balances in Jan 85. Adjust before using.
            # 'nyse_margin_debt_ratio',  # has an odd discontunuity in the credit balances in Jan 85. Adjust before using.
        ]

        ##########################################################################################################
        # Interactions between each x value and its previous values
        ##########################################################################################################
        for name in diff_x_names:
            for y in diff_quarters:
                diff_field_name = '{0}_{1}qtr_diff'.format(name, y)
                df[diff_field_name] = shift_x_quarters(df[name].pow(1./2.), y)
                # df[diff_field_name] = shift_x_quarters(df[name].apply(np.log, args=(1.5,)), y)
                x_names.append(diff_field_name)
        print('X Names Length: {0}'.format(len(x_names)))

        ##########################################################################################################
        # Value Imputation
        ##########################################################################################################
        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

        ##########################################################################################################
        # If even after imputation, some fields are empty, then you need to remove them
        ##########################################################################################################
        for n in x_names.copy():
            if df[n].isnull().any().any():
                print('Field [{0}] was still empty after imputation! Removing it!'.format(n))
                x_names.remove(n)

        ##########################################################################################################
        # Convert all x fields to EWMA/VWMA versions, to smooth craziness
        ##########################################################################################################
        print('Converting fields to EWMA/VWMA fields.')
        new_x_names = []
        for v in x_names:
            if use_vwma:
                new_field_name = v + '_vwma'
                if new_field_name not in df.columns.values:
                    res = vwma(df[v], mean_alpha=ewm_alpha)
                    if df[v].shape[0] != res.shape[0]:
                        print('========== VWMA RESULTS ==========')
                        print('Original: {rows}'.format(rows=df[v].shape))
                        print('Returned: {rows}'.format(rows=res.shape))
                        raise WTFException('THIS SHIT SHOULDN\'T HAPPEN')
                    df[new_field_name] = res
                    new_x_names.append(new_field_name)

            if use_ivwma:
                new_field_name = v + '_ivwma'
                if new_field_name not in df.columns.values:
                    res = vwma(df[v], mean_alpha=ewm_alpha, inverse=True)
                    if df[v].shape[0] != res.shape[0]:
                        print('========== VWMA RESULTS ==========')
                        print('Original: {rows}'.format(rows=df[v].shape))
                        print('Returned: {rows}'.format(rows=res.shape))
                        raise WTFException('THIS SHIT SHOULDN\'T HAPPEN')
                    df[new_field_name] = res
                    new_x_names.append(new_field_name)

            if use_ewma:
                new_field_name = v + '_ewma'
                if new_field_name not in df.columns.values:
                    df[new_field_name] = df[v].ewm(alpha=ewm_alpha).mean()
                    new_x_names.append(new_field_name)

        if len(new_x_names) > 0:
            x_names = new_x_names

        ##########################################################################################################
        # Trim all x fields to a threshold, expressed in terms of Stdev
        ##########################################################################################################
        if trim_stdevs > 0:
            print('Converting fields to trimmed fields.')
            new_x_names = []
            for v in x_names:
                new_field_name = v + '_trim'
                if new_field_name not in df.columns.values:
                    df[new_field_name] = trim_outliers(df[v], thresh=trim_stdevs)
                new_x_names.append(new_field_name)
            x_names = new_x_names

        ##########################################################################################################
        # Value Imputation
        ##########################################################################################################
        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

        ##########################################################################################################
        # Create and add any interaction terms
        ##########################################################################################################
        corr_x_names = None
        if correlation_type == 'level_1':
            print('Creating correlation interaction terms [{0}]'.format(correlation_type))
            df, corr_x_names = get_level1_correlations(df=df, x_names=x_names, ewm_alpha=ewm_alpha)
            print('X Names Length: {0}'.format(len(x_names)))

        ##########################################################################################################
        # Create interaction terms between the various x variables
        ##########################################################################################################
        print('Creating direct interaction terms [{0}]'.format(interaction_type))
        if interaction_type == 'all':
            df, new_x_names = get_all_interactions(df=df, x_names=x_names)
            x_names.extend(new_x_names)
            print('X Names Length: {0}'.format(len(x_names)))

        elif interaction_type == 'level_1':
            df, inter_x_names = get_level1_interactions(df=df, x_names=x_names)
            x_names.extend(inter_x_names)
            print('X Names Length: {0}'.format(len(x_names)))

        ################################################################################################################
        # DIMENSION REDUCTION: Remove any highly correlated items from the regression, to reduce issues with the model #
        ###############################################################################################################
        if (dimension_method is not None) and max_variables >= 0:
            if dimension_method == 'corr':
                df, x_names = reduce_vars_corr(df=df, field_names=x_names, max_num=max_variables,
                                               imputer=default_imputer)
            elif dimension_method == 'pca':
                df, x_names = reduce_vars_pca(df=df, field_names=x_names, max_num=max_variables)
            elif dimension_method == 'vif':
                df, x_names = remove_high_vif(X=df, max_num=max_variables)
            print('X Names Length: {0}'.format(len(x_names)))

        ##########################################################################################################
        # Create squared and squared-root versions of all the x fields
        ##########################################################################################################

        def create_transformed_field():
            lol = 1
            # Parallel(n_jobs=-1, verbose=-1)(delayed(variance_inflation_factor)(X.loc[:, colnames].values, ix) for ix in range(X.loc[:, colnames].shape[1]))


        if transform_vars:
            print('Creating squared and square root varieties of predictor variables')
            from scipy import stats
            operations = dict((
                              ('pow2', (math.pow, (2,))),
                              ('sqrt', (math.sqrt, None)),
                              # ('log2', (math.log2, None)),
                              # ('boxcox',(stats.boxcox, None)),
                               ))
            new_x_names = []
            for v in x_names:
                df[v] = df[v].astype(float)
                # boxcox_lambda = stats.boxcox_normmax(df[v].values)
                # operations['boxcox'] = (stats.boxcox,(boxcox_lambda,))
                for suffix, (op, var) in operations.items():

                    new_x_name = '{0}_{1}'.format(v, suffix)
                    df[new_x_name] = df[v].abs().apply(op, args=var) * df[v].apply(lambda x: -1 if x < 0 else 1)
                    new_x_names.append(new_x_name)
            x_names.extend(new_x_names)
            print('X Names Length: {0}'.format(len(x_names)))

        ##########################################################################################################
        # Add the x value correlations and trend variables (generated earlier) to the dataset
        ##########################################################################################################
        if corr_x_names:
            x_names.extend(corr_x_names)
            # IMPUTE VALUES!!!
            imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
            for n in x_names:
                df[n] = imputed_df[n]
            print('X Names Length: {0}'.format(len(x_names)))

        if trend_x_names:
            ##########################################################################################################
            # Impute the trend_x_names fields
            ##########################################################################################################
            imputed_df, trend_x_names = impute_if_any_nulls(df[trend_x_names], verbose=verbose)
            for n in trend_x_names:
                df[n] = imputed_df[n]

            ##########################################################################################################
            # If even after imputation, some fields are empty, then you need to remove them
            ##########################################################################################################
            for n in trend_x_names.copy():
                if df[n].isnull().any().any():
                    print('Field [{0}] was still empty after imputation! Removing it!'.format(n))
                    trend_x_names.remove(n)

            for v in trend_x_names:
                if use_vwma:
                    new_field_name = v + '_vwma'
                    if new_field_name not in df.columns.values:
                        df[new_field_name] = vwma(df[v], mean_alpha=ewm_alpha)
                        x_names.append(new_field_name)

                for v in trend_x_names:
                    if use_ivwma:
                        new_field_name = v + '_ivwma'
                        if new_field_name not in df.columns.values:
                            df[new_field_name] = vwma(df[v], mean_alpha=ewm_alpha, inverse=True)
                            x_names.append(new_field_name)

                if use_ewma:
                    new_field_name = v + '_ewma'
                    if new_field_name not in df.columns.values:
                        df[new_field_name] = df[v].ewm(alpha=ewm_alpha).mean()
                        x_names.append(new_field_name)

                # if use_vwma:
                #     new_field_name = v + '_vwma'
                # else:
                #     new_field_name = v + '_ewma'
                # if new_field_name not in df.columns.values:
                #     if use_vwma:
                #         df[new_field_name] = vwma(df[v], mean_alpha=ewm_alpha)
                #     else:
                #         df[new_field_name] = df[v].ewm(alpha=ewm_alpha).mean()

                # x_names.append(new_field_name)
            # IMPUTE VALUES!!!
            imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
            for n in x_names:
                df[n] = imputed_df[n]
            print('X Names Length: {0}'.format(len(x_names)))

        ################################################################################################################
        # DIMENSION REDUCTION: Remove any highly correlated items from the regression, to reduce issues with the model #
        ################################################################################################################
        if (dimension_method is not None) and max_variables >= 0:
            if dimension_method == 'corr':
                df, x_names = reduce_vars_corr(df=df, field_names=x_names, max_num=max_variables,
                                               imputer=default_imputer)
            elif dimension_method == 'pca':
                df, x_names = reduce_vars_pca(df=df, field_names=x_names, max_num=max_variables)
            elif dimension_method == 'vif':
                df, x_names = remove_high_vif(X=df, max_num=max_variables)
            print('X Names Length: {0}'.format(len(x_names)))

        ##########################################################################################################
        # Add x-variable for time since the last recession. Create y-variable for time until the next recession.
        ##########################################################################################################
        last_rec = -1
        for idx, period in enumerate(df.index.values):
            rec_val = df.get_value(period, recession_field_name)
            if rec_val == 1:
                for i, v in enumerate(df.iloc[last_rec:idx, :].index.values):
                    df.at[v, next_rec_field_name] = idx - last_rec - i
                last_rec = idx
                if idx > 0:
                    next_val = min(df[prev_rec_field_name].iloc[idx-1] - 1, 0)
                else:
                    next_val = -2
                df.at[period, prev_rec_field_name] = next_val
            elif rec_val == 0:
                df.at[period, prev_rec_field_name] = idx - last_rec
        x_names.append(prev_rec_field_name)

        # print(*df.index.values, sep='\n')

        ##########################################################################################################
        # IMPUTE SOME SHIT (make sure you don't have the timeuntil_next
        ##########################################################################################################
        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

        if do_predict_recession_quarters_method:
            imputed_df, _ = impute_if_any_nulls(df[x_names + [next_rec_field_name]], verbose=verbose)
            df[next_rec_field_name] = imputed_df[next_rec_field_name]

        ##########################################################################################################
        # Derive special predictor variables
        ##########################################################################################################
        if calc_trend_values:
            print('Deriving special predictor variables for y variable.')

            # Add x-variables for time since the last rise, and time since the last fall, in the SP500
            for x in [sp_field_name]:
                for stdev in [1.5, 2.25, 3]:
                    temp_df, new_x_names = get_diff_std_and_flags(df[x], x, stdev_qty=stdev)

                    sp500_qtr_since_fall = '{0}_time_since_{1}std_fall'.format(x, stdev)
                    df[sp500_qtr_since_fall] = time_since_last_true(temp_df[new_x_names[-1]])
                    x_names.append(sp500_qtr_since_fall)

                    sp500_qtr_since_rise = '{0}_time_since_{1}std_rise'.format(x, stdev)
                    df[sp500_qtr_since_rise] = time_since_last_true(temp_df[new_x_names[-2]])
                    x_names.append(sp500_qtr_since_rise)

                temp_df, new_x_names = get_trend_variables(df[x], x)

                # print(new_x_names)
                df = df.join(temp_df[new_x_names], how='inner')
                x_names.extend(new_x_names)

        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

        ################################################################################################################
        # DIMENSION REDUCTION: Remove any highly correlated items from the regression, to reduce issues with the model #
        ################################################################################################################
        if (dimension_method is not None) and max_variables >= 0:
            if dimension_method == 'corr':
                df, x_names = reduce_vars_corr(df=df, field_names=x_names, max_num=max_variables,
                                               imputer=default_imputer)
            elif dimension_method == 'pca':
                df, x_names = reduce_vars_pca(df=df, field_names=x_names, max_num=max_variables)
            elif dimension_method == 'vif':
                df, x_names = remove_high_vif(X=df, max_num=max_variables)
            print('X Names Length: {0}'.format(len(x_names)))

        ##########################################################################################################
        # VARIANCE REDUCTION: Remove any highly correlated fields and/or use pca to eliminate correlation.
        ##########################################################################################################
        if (correlation_method is not None) and max_correlation < 1. and dimension_method != 'pca':
            if correlation_method == 'corr':
                x_names = reduce_variance_corr(df=df, fields=x_names, max_corr_val=max_correlation, y=df[sp_field_name])
            elif correlation_method == 'pca':
                df, x_names = reduce_variance_pca(df=df, field_names=x_names, explained_variance=max_correlation)
            print('X Names Length: {0}'.format(len(x_names)))

    for n in x_names:
        if df[n].isnull().any().any():
            msg = 'Field "{0}" has null value!!!!'.format(n)
            print(msg)
            raise ValueError(msg)
        if any([np.isinf(v) for v in df[n].tolist()]):
            msg = 'Field "{0}" has null value!!!!'
            print(msg)
            raise ValueError(msg)
    print("== Final X Values for Modeling ===")
    print(*x_names, sep='\n')

    # print('===== Head =====\n', df.head(5))
    # print('===== Tail =====\n', df.tail(5))

    # print(df.head(5))
    # print(df.tail(5))

    # df_valid = df.loc[~train_mask, :]
    # predict_quarters_forward = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    print('Dumping final dataset (post-transformations) to pickle file [{0}]'.format(final_data_file))
    pickle_dump((df, x_names), final_data_file)

    # Predict the number of quarters until the next recession
    if do_predict_recession_quarters_method:
        for pred_dict, report_name in predict_recession_time(df=df,
                                                             x_names=x_names,
                                                             y_field_name=next_rec_field_name,
                                                             model_set=recession_quarters_models,
                                                             show_plot=show_plots):
            for k, v in pred_dict.items():
                df[k] = v

            # new_field_name = 'next_recession_{0}'.format(report_name)
            x_names.append([v for v in pred_dict.keys()][-1])

        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

    # RECESSION PREDICTIONS
    if do_predict_recession_flag:
        for yf in recession_flag_predict_quarters_forward:
            for pred_dict, report_name in predict_recession(df=df,
                                                            x_names=x_names,
                                                            y_field_name=recession_field_name,
                                                            quarters_forward=yf,
                                                            model_set=recession_flag_models,
                                                            show_plot=show_plots):
                for k, v in pred_dict.items():
                    df[k] = v

                new_field_name = 'recession_{0}'.format(report_name)
                for v in [v for v in list(pred_dict)][-2:]:
                    x_names.append(v)

        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

    if do_predict_returns:
        for yf in returns_predict_quarters_forward:
            for pred_dict, report_name in predict_returns(df=df,
                                                          x_names=x_names,
                                                          y_field_name=sp_field_name,
                                                          quarters_forward=yf,
                                                          model_set=returns_models,
                                                          comparison_asset_name=returns_comparison_asset_name,
                                                          show_plot=show_plots):
                for k, v in pred_dict.items():
                    df[k] = v
                x_names.append([v for v in pred_dict.keys()][-1])

                new_field_name = 'sp500_{0}'.format(report_name)

        imputed_df, x_names = impute_if_any_nulls(df[x_names], verbose=verbose)
        for n in x_names:
            df[n] = imputed_df[n]

    print('==== Execution Complete ====')
