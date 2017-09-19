import calendar
import math
import pickle
from datetime import datetime
from datetime import timedelta
# import fancyimpute
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import quandl
from Settings import Settings
import wbdata
import bls
import operator
# from matplotlib import rcParams
# import yahoo_finance
from joblib import Parallel, delayed
import itertools
from sklearn import feature_selection as sk_feat_sel
from statsmodels import robust
from fredapi import Fred
from matplotlib import pyplot as plt
from matplotlib.ticker import OldScalarFormatter
from sklearn.decomposition import TruncatedSVD
from inspect import isfunction
from pandas.compat import StringIO
import requests_cache
from Model_Builder import Model_Builder, ModelSet
from collections import OrderedDict
from typing import Union, Callable
import io
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests
import importlib

s = Settings(report_name='10YearS&P')

# Specify whether the data will be loaded from pickle files or recollected fresh from the interwebz
rawdata_from_file = True  # Whether to load the raw data (pre-transformations) from a pickle file
finaldata_from_file = False  # Whether to load the final data (post-transformations) from a pickle file

# Do you want to predict returns? Or Recessions?
do_predict_returns = True
returns_predict_quarters_forward = [2, 4]
do_predict_recessions = True
recession_predict_quarters_forward = [1]
do_predict_next_recession = False

# If you want to remove variables that don't meet a certain significance level, set this < 1. Requiring 95% = 5.0e-2.
selection_limit = 5.0e-2

interaction_type = 'level_1'  # Specify whether to derive pairwise interaction variables. Options: all, level_1, None
correlation_type = None  # Specify whether to derive pairwise EWM-correlation variables. Options: level_1, None
transform_vars = True
trim_vars = False
calc_trend_values = True
calc_std_values = True
calc_diff_from_trend_values = True

# VARIANCE REDUCTION. Either PCA Variance or Correlation Limits
correlation_method = None  # Use either None, 'corr' for correlations, or 'pca' for PCA
max_correlation = 0  # Options: 0 for 'auto' [0.99 for PCA, 0.80 for corr] or a float 0-1 for the amount of explained variance desired.

# DIMENSION REDUCTION. Either PCA Variance or Correlation Rankings
dimension_method = 'corr'  # Use either None, 'corr' for correlations, or 'pca' for PCA
max_variables = 0.65  # Options: 0 for 'auto' [n_obs ^ (1/2)] or an integer for a specific number of variables.

# Variables specifying what kinds of predictions to run, and for what time period
start_dt = '1920-01-01'
end_dt = datetime.today().strftime('%Y-%m-%d')
train_pct = 0.8  # Defines the train/test split
real = False
sp_field_name = 'sp500'
recession_field_name = 'recession_usa'
prev_rec_field_name = 'recession_usa_time_since_prev'
next_rec_field_name = 'recession_usa_time_until_next'
verbose = False
fred = Fred(api_key='b604ef6dcf19c48acc16461e91070c43')
ewm_alpha = 0.125  # Halflife for EWM calculations. 4.2655 corresponds to a 0.125 weight.
default_imputer = 'knnimpute'  # 'fancyimpute' or 'knnimpute'. knnimpute is generally much faster, if less ideal.
stack_include_preds = False
final_include_data = False

if do_predict_next_recession:
    initial_models = ['rfor_r','gbr','elastic_net','knn_r','svr']
    if max_correlation < 1. or max_variables == 0:
        initial_models.extend(['neural_r','gauss_proc_r'])

    final_models = ['elastic_net_stacking']

    next_recession_models = ModelSet(final_models=final_models, initial_models=initial_models)

if do_predict_recessions:
    initial_models=['logit','etree_c','nearest_centroid','gbc','bernoulli_nb','svc','rfor_c'] # pass_agg_c
    # initial_models=['logit','etree_c']
    if max_correlation < 1. or max_variables == 0:
        initial_models.extend(['gauss_proc_c', 'neural_c'])

    # BEST MODELS: logit, svc, sgd_c, neural_c, gauss_proc_c
    # Overall best model: svc???
    final_models=['logit','svc']
    if (not final_include_data) or max_correlation < 1. or max_variables == 0:
        final_models.extend(['gauss_proc_c', 'neural_c'])
    final_models = 'gauss_proc_c'

    # initial_models=['logit','etree_c','nearest_centroid','gbc','bernoulli_nb','svc','pass_agg_c','gauss_proc_c']
    recession_models = ModelSet(final_models=final_models, initial_models=initial_models)

if do_predict_returns:
    initial_models = ['rfor_r','gbr','pass_agg_r','elastic_net','knn_r','ridge_r','svr']
    if max_correlation < 1. or max_variables == 0:
        initial_models.extend(['gauss_proc_r'])

    final_models = ['elastic_net_stacking','ridge_r','linreg','svr']
    if (not final_include_data) or max_correlation < 1. or max_variables == 0:
        final_models.extend(['gauss_proc_r'])

    returns_models = ModelSet(final_models=final_models, initial_models=initial_models)

if real:
    sp_field_name += '_real'

#use the list to filter the local namespace
# safe_func_list = ['first', 'last', 'mean']
# safe_funcs = dict([(k, locals().get(k, None)) for k in safe_func_list])

def make_qtrly(s: pd.Series, t: str='mean') -> pd.Series:
    s.index = pd.DatetimeIndex(s.index.values, dtype=datetime.date)
    s.index.freq = s.index.inferred_freq

    # print(s)

    if t == 'mean':
        s = s.resample('1Q').mean().astype(np.float64)
    elif t == 'first':
        s = s.resample('1Q').first().astype(np.float64)
    elif t == 'last':
        s = s.resample('1Q').last().astype(np.float64)

    # Conform everything to the end of the quarter
    idx = s.index
    for i, v in enumerate(idx):
        v.replace(month=math.ceil(v.month/3)*3)
        v.replace(day=calendar.monthrange(v.year, v.month)[-1])
    s.index = idx
    
    # s.index = s.index + pd.Timedelta(3, unit='M') - pd.Timedelta(1, unit='d')

    # s.index = pd.to_datetime([d + relativedelta(days=1) for d in s.index])
    # s.index.freq = s.index.inferred_freq

    # I wanted to make this function more dynamic and eliminate the if/else bullshit, with the below line (which failed)
    # s = s.resample('3MS').apply(eval(t + '(self)', {"__builtins__": None}, safe_funcs)).astype(np.float64)

    # print(s)
    return s

def get_closes(d: list, fld_names: list=list(('Close', 'col3'))) -> pd.Series:
    index_list = []
    val_list = []
    date_names = ['Date','col0']
    for i, v in enumerate(d):
        for name in date_names:
            if name in v:
                index_list.append(v[name])
                break
        else:
            raise ValueError("Couldn't find any possible vals {0} in row {1} the dataset".format(date_names, i))

        for name in fld_names:
            if name in v:
                val_list.append(v[name])
                break
        else:
            raise ValueError("Couldn't find any possible vals {0} in row {1} of the dataset".format(fld_names, i))
    return pd.Series(val_list, index=index_list)

# d = Pandas Dataframe
# ys = [ [cols in the same y], [cols in the same y], [cols in the same y], .. ]
# invert = [[True/False to invert y axis], [True/False to invert y axis], [True/False to invert y axis], ..]
def chart(d, ys, invert, log_scale, save_name=None, title=None):

    from itertools import cycle
    fig, ax = plt.subplots()

    axes = [ax]
    for y in ys[1:]:
        # Twin the x-axis twice to make independent y-axes.
        axes.append(ax.twinx())

    extra_ys = len(axes[2:])

    # Make some space on the right side for the extra y-axes.
    right_additive = 0
    if extra_ys>0:
        temp = 0.85
        if extra_ys<=2:
            temp = 0.75
        elif extra_ys<=4:
            temp = 0.6
        if extra_ys>5:
            print('you are being ridiculous')
        fig.subplots_adjust(right=temp)
        right_additive = (0.98-temp)/float(extra_ys)
    # Move the last y-axis spine over to the right by x% of the width of the axes
    i = 1.
    for ax in axes[2:]:
        ax.spines['right'].set_position(('axes', 1.+right_additive*i))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        ax.yaxis.set_major_formatter(OldScalarFormatter())
        i +=1.
    # To make the border of the right-most axis visible, we need to turn the frame
    # on. This hides the other plots, however, so we need to turn its fill off.

    cols = []
    lines = []
    line_styles = cycle(['-','-','-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>',
               '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])
    colors = cycle([p['color'] for p in list(matplotlib.rcParams['axes.prop_cycle'])])
    for i, (ax, y) in enumerate(zip(axes, ys)):
        ls=next(line_styles)
        if len(y)==1:
            col = y[0]
            cols.append(col)
            color = next(colors)
            lines.append(ax.plot(d[col], linestyle=ls, label=col, color=color))
            ax.set_ylabel(col,color=color)
            #ax.tick_params(axis='y', colors=color)
            ax.spines['right'].set_color(color)
        else:
            for col in y:
                color = next(colors)
                lines.append(ax.plot(d[col], linestyle=ls, label=col, color=color))
                cols.append(col)
            ax.set_ylabel(' // '.join(y))
            # ax.tick_params(axis='y')
        if invert[i]:
            ax.invert_yaxis()
        if log_scale[i]:
            ax.set_yscale('log')
    axes[0].set_xlabel(d.index.name)
    lns = lines[0]
    for l in lines[1:]:
        lns +=l
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=2)

    if title:
        plt.title(title)
    else:
        plt.title(list(itertools.chain(*ys))[0])
    if save_name:
        fig.savefig('{0}/{1}'.format(s.get_reports_dir(), save_name))
    print("Showing Plot...")
    plt.show()


def get_obj_name(o) -> str:
    return [k for k, v in locals().items() if v is o][0]

def reverse_enumerate(l):
   for index in reversed(range(len(l))):
      yield index, l[index]

def predict_returns(df: pd.DataFrame,
                    x_names: list,
                    y_field_name: str,
                    model_set: ModelSet,
                    quarters_forward: int)\
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

    train_mask = ~df[forward_y_field_name].isnull()

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

    report_name = 'returns_{0}qtr_{1}'.format(quarters_forward, model_set)
    m = Model_Builder(df,
                      x_fields=x_fields,
                      y_field=y_field,
                      model_type=model_set,
                      report_name=report_name,
                      show_model_tests=True,
                      retrain_model=True,
                      selection_limit=selection_limit,
                      predict_all=True,
                      verbose=verbose,
                      train_pct=train_pct,
                      random_train_test=False,
                      stack_include_preds=stack_include_preds,
                      final_include_data=final_include_data,
                      correlation_max=0.95,
                      use_test_set=True,
                      use_sparse=False
                      )

    for df, final_model_name in m.predict():

        forward_y_field_name_pred = 'pred_' + forward_y_field_name
        #################################

        invest_amt_per_qtr = 5000
        df['invest_strat_cash'] = invest_amt_per_qtr
        df['invest_strat_basic'] = invest_amt_per_qtr
        df['invest_strat_mixed'] = invest_amt_per_qtr
        df['invest_strat_equity'] = invest_amt_per_qtr
        df['invest_strat_tsy'] = invest_amt_per_qtr

        mean_y_field_return = df[forward_y_field_name].mean()

        for i in range(df.shape[0] - quarters_forward):
            df.ix[i + (quarters_forward), 'invest_strat_cash'] += df.ix[i, 'invest_strat_cash']

            if mean_y_field_return >= df.ix[i, 'tsy_10yr_yield']:
                df.ix[i + (quarters_forward), 'invest_strat_basic'] += df.ix[i, 'invest_strat_basic'] * (1 + df.ix[
                    i, forward_y_field_name]) ** quarters_forward
            else:
                df.ix[i + (quarters_forward), 'invest_strat_basic'] += df.ix[i, 'invest_strat_basic'] * (1 + df.ix[
                    i, 'tsy_10yr_yield']) ** quarters_forward

            if df.ix[i, forward_y_field_name_pred] >= df.ix[i, 'tsy_10yr_yield']:
                df.ix[i + (quarters_forward), 'invest_strat_mixed'] += df.ix[i, 'invest_strat_mixed'] * (1 + df.ix[
                    i, forward_y_field_name]) ** quarters_forward
            else:
                df.ix[i + (quarters_forward), 'invest_strat_mixed'] += df.ix[i, 'invest_strat_mixed'] * (1 + df.ix[
                    i, 'tsy_10yr_yield']) ** quarters_forward

            df.ix[i + (quarters_forward), 'invest_strat_equity'] += df.ix[i, 'invest_strat_equity'] * (1 + df.ix[
                i, forward_y_field_name]) ** quarters_forward

            df.ix[i + (quarters_forward), 'invest_strat_tsy'] += df.ix[i, 'invest_strat_tsy'] * (1 + df.ix[
                i, 'tsy_10yr_yield']) ** quarters_forward

        df['total_strat_cash'] = df['invest_strat_cash'][train_mask].rolling(window=(quarters_forward)).sum()
        df['total_strat_basic'] = df['invest_strat_basic'][train_mask].rolling(window=(quarters_forward)).sum()
        df['total_strat_mixed'] = df['invest_strat_mixed'][train_mask].rolling(window=(quarters_forward)).sum()
        df['total_strat_equity'] = df['invest_strat_equity'][train_mask].rolling(window=(quarters_forward)).sum()
        df['total_strat_tsy'] = df['invest_strat_tsy'][train_mask].rolling(window=(quarters_forward)).sum()

        final_return_strat_cash = df['total_strat_cash'][train_mask].iloc[-1]
        final_return_strat_basic = df['total_strat_basic'][train_mask].iloc[-1]
        final_return_strat_mixed = df['total_strat_mixed'][train_mask].iloc[-1]
        final_return_strat_equity = df['total_strat_equity'][train_mask].iloc[-1]
        final_return_strat_tsy = df['total_strat_tsy'][train_mask].iloc[-1]

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
        df[forward_y_field_name_pred_fut] = df[forward_y_field_name_pred].astype(np.float16)
        df[forward_y_field_name_pred_fut][train_mask] = np.nan

        print(max(df.index))
        from dateutil.relativedelta import relativedelta
        df = df.reindex(
            pd.DatetimeIndex(start=df.index.min(),
                             end=max(df.index) + relativedelta(months=quarters_forward*3),
                             freq='1Q',
                             dtype=datetime.date))

        y_field_name_pred = '{0}_pred'.format(y_field_name)
        df[y_field_name_pred] = (df[y_field_name] * df[forward_y_field_name_pred].add(1).pow(quarters_forward)).shift(quarters_forward)

        report_name = 'returns_{0}qtr_{1}_{2}'.format(quarters_forward, model_set, final_model_name)

        # if quarters_forward in [10]:
        chart(df
              , ys=[[forward_y_field_name, forward_y_field_name_pred, 'tsy_10yr_yield'], [y_field_name, y_field_name_pred]]
              , invert=[False, False]
              , log_scale=[False, True]
              , save_name='_'.join([forward_y_field_name_pred, report_name, str(quarters_forward)])
              , title='_'.join([y_field_name, report_name]))

        yield (OrderedDict(((forward_y_field_name, df[forward_y_field_name]),
            (forward_y_field_name_pred, df[forward_y_field_name_pred]),
            (y_field_name_pred, df[y_field_name_pred]))),
            report_name)



def predict_recession(df: pd.DataFrame
                      , x_names: list
                      , y_field_name: str
                      , quarters_forward: int
                      , model_set: ModelSet)\
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

    m = Model_Builder(df,
                      x_fields=x_fields,
                      y_field=y_field,
                      model_type=model_set,
                      report_name=report_name,
                      show_model_tests=True,
                      retrain_model=False,
                      selection_limit=selection_limit,
                      predict_all=True,
                      verbose=verbose,
                      train_pct=train_pct,
                      random_train_test=False,
                      pca_explained_var=1.0,
                      stack_include_preds=stack_include_preds,
                      final_include_data=final_include_data,
                      use_test_set=True,
                      correlation_method='corr',
                      correlation_max=0.95,
                      use_sparse=False
                      # cross_val_iters=(20,20),
                      )

    for df, final_model_name in m.predict():

        forward_y_field_name_pred = 'pred_' + forward_y_field_name
        #################################

        print(max(df.index))
        from dateutil.relativedelta import relativedelta
        df = df.reindex(
            pd.DatetimeIndex(start=df.index.min(),
                             end=max(df.index) + relativedelta(months=quarters_forward*3),
                             freq='1Q',
                             dtype=datetime.date))

        y_field_name_pred = '{0}_pred'.format(y_field_name)
        df[y_field_name_pred] = df[forward_y_field_name_pred].shift(quarters_forward)
        chart_y_field_names = [y_field_name, y_field_name_pred]

        y_pred_names = m.new_fields
        y_prob_field_name = '{0}_prob_1.0'.format(forward_y_field_name)
        if y_prob_field_name in y_pred_names:
            df[y_prob_field_name] = df[y_prob_field_name].shift(quarters_forward)
            chart_y_field_names += [y_prob_field_name]

        report_name = 'recession_{0}qtr_{1}_{2}'.format(quarters_forward, model_set, final_model_name)
        # if quarters_forward in [10]:
        chart(df
              , ys=[['sp500'], chart_y_field_names]
              , invert=[False, False]
              , log_scale=[True, False]
              , save_name='_'.join([forward_y_field_name_pred, report_name])
              , title='_'.join([y_field_name, report_name]))

        yield (OrderedDict((
                            (forward_y_field_name, df[forward_y_field_name].astype(float)),
                            (forward_y_field_name_pred, df[forward_y_field_name_pred].astype(float)),
                            (y_field_name_pred, df[y_field_name_pred].astype(float)),
                            (y_prob_field_name, df[y_prob_field_name].astype(float)),
                            )),
               report_name)


def predict_recession_time(df: pd.DataFrame,
                           x_names: list,
                           y_field_name: str,
                           model_set: ModelSet)\
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

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Generate all different combinations of p, q and q triplets
    # Generate all different combinations of seasonal p, q and q triplets

    p = d = q = range(1, 2) # Change to 0, 2
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
    start_pred = pd.DatetimeIndex(df.loc[train_mask].index.shift(1,'Q'),
                                  dtype=datetime.date)[-1]
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
    chart(df
          , ys=[['sp500'], [y_field_name, y_field_name_pred, 'tsy_10yr_yield']]
          , invert=[False, False]
          , log_scale=[True, False]
          , save_name='_'.join([y_field_name_pred, report_name])
          , title='_'.join([y_field_name, report_name]))

    yield (OrderedDict((
                        (y_field_name, df[y_field_name].astype(float)),
                        (y_field_name_pred, df[y_field_name_pred].astype(float)),
                        )),
           report_name)


def decimal_to_date(d: str):
    year = int(float(d))
    frac = float(d) - year
    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * frac)
    return result

# lol = decimal_to_date('2017.3213421')
# print(lol)

class data_source:
    def __init__(self, code: str, provider: Union[str, Callable], rerun: bool=False):
        self.code = code
        self.data = pd.Series()
        self.rerun = rerun

        if provider in ('fred', 'yahoo', 'quandl', 'schiller', 'eod_hist', 'bls', 'worldbank'):
            self.provider = provider
        elif isfunction(provider):
            self.provider = provider
        else:
            raise ValueError('Unrecognized data source provider passed to new data source.')

    def set_data(self, data: pd.DataFrame):
        self.data = data

    def collect_data(self):
        global start_dt, end_dt
        if isfunction(self.provider):
            if self.code:
                self.data = self.provider(self.code)
            else:
                self.data = self.provider(self.code)
        elif self.provider == 'fred':
            self.data = fred.get_series(self.code
                                        , observation_start=start_dt
                                        , observation_end=end_dt)
        elif self.provider == 'eod_hist':
            url = 'https://eodhistoricaldata.com/api/eod/{0}'.format(self.code)
            params = {'api_token': '599dc44361b10'}
            expire_after = timedelta(days=1).total_seconds()
            session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)
            r = session.get(url, params=params)
            if r.status_code != requests.codes.ok:
                session = requests.Session()
                r = session.get(url, params=params)
            if r.status_code == requests.codes.ok:
                df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
                self.data = df['Close']
            else:
                raise Exception(r.status_code, r.reason, url)

        elif self.provider == 'schiller':
            import csv
            import io
            import urllib.request
            import bs4
            url = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
            webpage = requests.get(url, stream=True)
            self.data = pd.read_excel(io.BytesIO(webpage.content), 'Data', header=7, skip_footer=1)
            self.data.index = self.data['Date'].apply(lambda x: datetime.strptime(str(x).format(x, '4.2f'), '%Y.%m'))
            self.data = self.data[self.code]
            print(self.data.tail(5))
            lol = 1

        elif self.provider == 'quandl':
            self.data = quandl.get(self.code
                                   , authtoken="xg_fvD6FLD_qzg2Mc5z-"
                                   , collapse="quarterly"
                                   , start_date=start_dt
                                   , end_date=end_dt)['Value']
        elif self.provider == 'bls':
            self.data = bls.get_series([self.code]
                                , startyear=datetime.strptime(start_dt, '%Y-%m-%d').year
                                , endyear=datetime.strptime(end_dt, '%Y-%m-%d').year
                                , key='3d75f024d5f64d189e5de4b7cbb99730'
                                )
        elif self.provider == 'worldbank':
            self.data = wbdata.get_data(self.code,
                                        country='US',
                                        data_date=(datetime.strptime(start_dt, '%Y-%m-%d'),
                                                   datetime.strptime(end_dt, '%Y-%m-%d')),
                                        convert_date=True,
                                        pandas=True,
                                        keep_levels=False
                                   )
            print(self.data)
            lol=1
        print("Collected data for [{0}]".format(self.code))


def shift_x_quarters(s: pd.Series, y: int):
    return s / s.shift(y)


def impute_if_any_nulls(impute_df: pd.DataFrame, imputer: str=default_imputer):
    if impute_df.isnull().any().any():
        print('Running imputation')
        try:
            if imputer == 'knnimpute':
                raise ValueError('knnimpute requested')
            imputer = importlib.import_module("fancyimpute")
            solver = imputer.MICE(init_fill_method='random') # mean, median, or random
            # solver = fancyimpute.NuclearNormMinimization()
            # solver = fancyimpute.MatrixFactorization()
            # solver = fancyimpute.IterativeSVD()
            impute_df = solver.complete(impute_df.values)
        except (ImportError, ValueError) as e:
            imputer = importlib.import_module("knnimpute")
            impute_df = imputer.knn_impute_few_observed(impute_df.values, missing_mask=impute_df.isnull().values, k=5, verbose=verbose)
        # df = solver.complete(df.values)
    return impute_df



def get_operator_fn(op):
    return {
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '/' : operator.truediv,
        '%' : operator.mod,
        '^' : operator.xor,
        }[op]


def permutations_with_replacement(n, k):
    for p in itertools.product(n, repeat=k):
        yield p


# Construct level 1 interactions between all x-variables
def get_level1_interactions(df: pd.DataFrame, x_names: list, min: float=0.1, max: float=0.9):
    print('Adding interaction terms.')
    new_x_names = []
    for k, v in enumerate(x_names[:-1]):
        for v1 in x_names[k + 1:]:
            corr = np.corrcoef(df[v], df[v1])[0][1]
            if min <= abs(corr) <= max:
                # Interactions between two different fields generated through multiplication
                interaction_field_name = '{0}_*_{1}'.format(v, v1)
                if interaction_field_name not in df.columns.values:
                    df[interaction_field_name] = df[v] * df[v1]
                new_x_names.append(interaction_field_name)

                # Interactions between two different fields, generated through division
                interaction_field_name = '{0}_/_{1}'.format(v, v1)
                if interaction_field_name not in df.columns.values:
                    df[interaction_field_name] = df[v] / df[v1].replace({0: np.nan})
                new_x_names.append(interaction_field_name)

                # Interactions between two different fields, generated through exponentiall weighted correlations
                # interaction_field_name = '{0}_corr_{1}'.format(v, v1)
                # df[interaction_field_name] = df[v].ewm(alpha=ewm_alpha).corr(other=df[v1])
                # new_x_names.append(interaction_field_name)

    return df, new_x_names



def get_diff_std_and_flags(s: pd.Series
                           , field_name: str
                           , alpha: float=ewm_alpha
                           , stdev_qty: float=2.)\
        -> (pd.DataFrame, list):

    new_name_ewma = field_name + '_ewma'
    new_name_std = field_name + '_std'
    new_name_prefix = field_name + '_val_to_ewma'
    new_name_diff = new_name_prefix + '_diff'
    new_name_diff_std = new_name_prefix + '_diff_std'
    new_name_add_std = new_name_prefix + '_add_{0}std'.format(stdev_qty)
    new_name_sub_std = new_name_prefix + '_sub_{0}std'.format(stdev_qty)
    new_name_trend_rise = new_name_prefix + '_trend_rise_flag_{0}std'.format(stdev_qty)
    new_name_trend_fall = new_name_prefix + '_trend_fall_flag_{0}std'.format(stdev_qty)

    s_ewma = s.ewm(alpha=alpha).mean()
    s_std = s.ewm(alpha=alpha).std(bias=True).shift(1)
    s_diff = s - s_ewma
    s_diff_std = s_diff.shift(1).ewm(alpha=alpha).std(bias=True)
    s_ewma_add_std = s_ewma.shift(1) + (stdev_qty * s_diff_std)
    s_ewma_sub_std = s_ewma.shift(1) - (stdev_qty * s_diff_std)
    s_ewma_sub_std = s_ewma_sub_std.clip(lower=0)
    d = pd.DataFrame(index=s.index)
    d[new_name_trend_rise] = s.values > s_ewma_add_std.values
    d[new_name_trend_fall] = s.values < s_ewma_sub_std.values

    return d, [new_name_diff, new_name_diff_std, new_name_add_std, new_name_sub_std, new_name_trend_rise,
               new_name_trend_fall]


def get_trend_variables(s: pd.Series,
                        field_name: str,
                        alpha: float=0.25)\
        -> (pd.DataFrame, list):

    d = s.to_frame(name=field_name)

    ema_df = pd.DataFrame()
    new_name_trend_strength = field_name + '_trend_strength'
    new_name_trend_strength_weighted = field_name + '_trend_strength_weighted'

    range_len = 17
    index_range = range(1, range_len)
    for exp in index_range:
        h1 = alpha*(1/(exp))
        h2 = alpha*(1/(exp+1))
        ema1 = d[field_name].ewm(alpha=h1).mean()
        ema2 = d[field_name].ewm(alpha=h2).mean()
        ema_df[str(exp)] = ema1 / ema2
        # ema_df[col_name] = (ema_df[col_name] / ema_df[col_name].shift(1)).fillna(1)
    # for exp in range(1, 17):
    #     ema_df[str(exp)] = d[field_name].rolling(exp*2).mean()
    #     ema_df[str(exp)] = (ema_df[str(exp)] / ema_df[str(exp)].shift(1)).fillna(1)

    # weights = [2**(v-1) for v in index_range]
    weights = [v for v in index_range]
    weights = pd.Series(list(map(lambda x: x / sum(weights), weights)), index=ema_df.columns.values)
    d[new_name_trend_strength_weighted] = ema_df.subtract(-1).dot(weights)
    neg = np.sum((ema_df.values < 1), axis=1)
    pos = np.sum((ema_df.values > 1), axis=1)
    d[new_name_trend_strength] = ((pos - neg) + range_len) / (2*range_len)  # creates an index, 0-1, of trend strength
    return d, [new_name_trend_strength, new_name_trend_strength_weighted]


def get_std_from_ewma(s: pd.Series, field_name: str, alpha: float=ewm_alpha/4) -> (pd.DataFrame, list):

    d = s.to_frame(name=field_name)

    new_name_ewma = field_name + '_ewma'
    new_name_std = field_name + '_std'
    new_name_std_from_ewma = field_name + '_std_from_ewma'

    d[new_name_ewma] = d[field_name].ewm(alpha=alpha).mean()
    d[new_name_std] = d[field_name].ewm(alpha=alpha).std(bias=True)
    d[new_name_std_from_ewma] = (d[field_name] - d[new_name_ewma]) / d[new_name_std]
    return d, [new_name_std_from_ewma]


def get_diff_from_trend(s: pd.Series, field_name: str) -> (pd.DataFrame, list):

    N = len(s.values)
    from sklearn import linear_model
    n = pd.Series([v+2 for v in range(len(s.values))], index=s.index)
    logn = np.log(n)

    # fit linear model
    linmod = linear_model.LinearRegression()
    x = np.reshape(n, (N, 1))
    y = s
    linmod.fit(x, y)
    linmod_rsquared = linmod.score(x, y)
    m = linmod.coef_[0]
    c = linmod.intercept_
    linear = c + (n * m)

    # fit log-log model
    loglogmod = linear_model.LinearRegression()
    x = np.reshape(logn, (N, 1))
    y = np.log(s)
    loglogmod.fit(x, y)
    loglogmod_rsquared = loglogmod.score(x, y)
    m = loglogmod.coef_[0]
    c = loglogmod.intercept_
    polynomial = math.exp(c) * np.power(n, m)

    # fit log model
    logmod = linear_model.LinearRegression()
    x = np.reshape(n, (N, 1))
    y = np.log(s)
    logmod.fit(x, y)
    logmod_rsquared = logmod.score(x, y)
    m = logmod.coef_[0]
    c = logmod.intercept_
    exponential = np.exp(n * m) * math.exp(c)

    if False:
        # plot results
        plt.subplot(1, 1, 1)
        plt.plot(n, s, label='series {0}'.format(field_name), lw=2)

        # linear model
        m = linmod.coef_[0]
        c = linmod.intercept_
        plt.plot(n, linear, label='$t={0:.1f} + n*{{{1:.1f}}}$ ($r^2={2:.4f}$)'.format(c, m, linmod_rsquared),
                 ls='dashed', lw=3)

        # log-log model
        m = loglogmod.coef_[0]
        c = loglogmod.intercept_
        plt.plot(n, polynomial, label='$t={0:.1f}n^{{{1:.1f}}}$ ($r^2={2:.4f}$)'.format(math.exp(c), m, loglogmod_rsquared),
                 ls='dashed', lw=3)

        # log model
        m = logmod.coef_[0]
        c = logmod.intercept_
        plt.plot(n, exponential, label='$t={0:.1f}e^{{{1:.1f}n}}$ ($r^2={2:.4f}$)'.format(math.exp(c), m, logmod_rsquared),
                 ls='dashed', lw=3)

        # Show the plot results
        plt.legend(loc='upper center', prop={'size': 16}, borderaxespad=0., bbox_to_anchor=(0.5, 1.25))
        plt.show()

    max_rsq = round(max(linmod_rsquared, logmod_rsquared, loglogmod_rsquared),2)
    if linmod_rsquared + 0.1 >= max_rsq:
        trend = linear
    elif loglogmod_rsquared + 0.05 >= max_rsq:
        trend = polynomial
    else:
        trend = exponential


    new_name_vs_trend = field_name + '_vs_trend'
    d = pd.DataFrame(s / trend, columns=[new_name_vs_trend])
    return d, [new_name_vs_trend]


def time_since_last_true(s: pd.Series) -> pd.Series:
    s.iloc[0] = prev_val = s.value_counts()[False] / 2 / s.value_counts()[True]
    for i, v in list(s.iteritems())[1:]:
        if v:
            s.set_value(i, 0)
        else:
            s.set_value(i, prev_val + 1)
        prev_val = s.get_value(i)
    return s


def get_level1_correlations(df: pd.DataFrame, x_names: list, top_n: int=-1):
    # Interactions between two different fields, generated through exponentiall weighted correlations
    print('Adding correlation terms')
    new_x_names = []
    d = dict()
    for i, v in enumerate(x_names[:-1]):
        for v1 in x_names[i + 1:]:
            # Interactions between two different fields, generated through exponentiall weighted correlations
            interaction_field_name = '{0}_corr_{1}'.format(v, v1)
            if interaction_field_name not in df.columns.values:
                # s = pd.ewmcorr(df[v], df[v1], alpha=ewm_alpha)
                s = df[v].ewm(alpha=ewm_alpha).corr(other=df[v1]).round(4)
                # s = df[v].apply(lambda x: x.fillna(0).ewm(alpha=ewm_alpha).corr(other=df[v1]))
            else:
                s = df[interaction_field_name]
            d[interaction_field_name] = (s, s.var())

    if top_n > len(d):
        top_n = -1

    for name, vals in sorted(d.items(), key=lambda x: x[1][1], reverse=True)[:top_n]:
        df[name] = vals[0]
        new_x_names.append(name)

    return df, new_x_names


# Construct all possible interactions between x-variables
def get_all_interactions(new_names: list, curr_name: str=None, series: pd.Series=None):
    if new_names:
        next_name = new_names[0]
        if not curr_name:
            get_all_interactions(new_names=new_names[1:], curr_name=next_name, series=df[next_name])
        else:
            if abs(np.corrcoef(series, df[next_name])[0][1]) <= 0.3:
                for op in ['*', '/']:
                    next_curr_name = '{0}_{1}_{2}'.format(curr_name, op, next_name)
                    next_series = get_operator_fn(op)(series, df[next_name].replace({0: np.nan}))
                    new_x_names.append(next_curr_name)
                    df[next_curr_name]= next_series
                    print("Adding interaction term: {0}".format(next_curr_name))
                    get_all_interactions(new_names=new_names[1:], curr_name=next_curr_name, series=next_series)
            else:
                get_all_interactions(new_names=new_names[1:], curr_name=curr_name, series=series)
        get_all_interactions(new_names=new_names[1:])


def trim_outliers(y: pd.Series, thresh=4):
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    mask = ~np.isnan(y)
    med = np.median(y[mask])
    mad = robust.mad(y[mask])
    min_thresh = med - (mad * thresh)
    max_thresh = med + (mad * thresh)
    y[y > max_thresh] = max_thresh
    y[y < min_thresh] = min_thresh
    return y


def reduce_vars_corr(df: pd.DataFrame, field_names: list, max_num: float):
    num_vars = len(field_names)-1
    print('Current vars:  {0}'.format(num_vars))
    if not max_num or max_num < 1:
        if max_num == 0:
            max_num = 0.75
        max_num = int(np.power(df.shape[0], max_num))

    print('Max allowed vars: {0}'.format(max_num))

    if num_vars > max_num:

        # Creates Correlation Matrix
        corr_matrix = df.loc[:, field_names].corr()

        max_corr = [(fld, corr_matrix.iloc[i+1, :(i)].max()) for i, fld in reverse_enumerate(field_names[1:])]
        max_corr.sort(key=lambda tup: tup[1])

        return_x_vals = [fld for fld, corr in max_corr[:max_num]]
        print('Number of Remaining Fields: {0}'.format(len(return_x_vals)))
        print('Remaining Fields: {0}'.format(return_x_vals))
        return return_x_vals
    return field_names

def reduce_vars_pca(df: pd.DataFrame, field_names: list, max_num: Union[int,float]=None):
    num_vars = len(field_names)-1
    print('Current vars: {0}'.format(num_vars))
    if not max_num or max_num == 0:
        max_num = int(np.power(df.shape[0], 0.7))

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


def reduce_variance_corr(df: pd.DataFrame, fields: list, max_corr_val: float, y: Union[list, pd.Series, np.ndarray]):
    print('Removing one variable for each pair of variables with correlation greater than [{0}]'.format(max_corr_val))
    # Creates Correlation Matrix and Instantiates
    corr_matrix = df[fields].astype(float).corr(method='pearson')
    drop_cols = set()
    if max_corr_val <= 0.:
        max_corr_val = 0.8

    # Determine the p-values of the dataset and when a field must be dropped, prefer the field with the higher p-value
    if len(np.unique(y)) == 2:
        scores, p_vals = sk_feat_sel.f_classif(df[fields], y)
    else:
        scores, p_vals = sk_feat_sel.f_regression(df[fields], y)

    # Iterates through Correlation Matrix Table to find correlated columns
    for i, v in enumerate(fields[:-1]):
        i2, c = sorted([(i2+1, v2) for i2, v2 in enumerate(corr_matrix.iloc[i, i+1:])], key=lambda tup: tup[1])[-1]

        if c > max_corr_val:
            if p_vals[i] <= p_vals[i2]:
                drop_cols.add(fields[i2])
            else:
                drop_cols.add(v)

    return_x_vals = [v for v in fields if v not in list(drop_cols)]
    print('=== Drop of Highly-Correlated Variables is Complete ===')
    print('Dropped Fields [{0}]: {1}'.format(len(drop_cols), list(drop_cols)))
    print('Remaining Fields [{0}]: {1}'.format(len(return_x_vals), return_x_vals))
    return return_x_vals


def reduce_variance_pca(df: pd.DataFrame, field_names: list, explained_variance: float):
    print("Conducting PCA and pruning components above the desired explained variance ratio")
    max_components = len(field_names) - 1
    if explained_variance <= 0:
        explained_variance = 0.99

    pca_model = TruncatedSVD(n_components=max_components, random_state=555)
    x_results = pca_model.fit_transform(df.loc[:, field_names]).T
    # print(pca_model.components_)
    print('PCA explained variance ratios.')
    print(pca_model.explained_variance_ratio_)

    x_names_pca = []
    sum_variance = 0.
    for idx, var in enumerate(pca_model.explained_variance_ratio_):
        sum_variance += var
        pca_name = 'pca_{0}'.format(idx)
        df[pca_name] = x_results[idx]
        x_names_pca.append(pca_name)
        if sum_variance > explained_variance:
            break

    print('Explained variance retained: {0:.2f}'.format(sum_variance))
    print('Number of PCA Fields: {0}'.format(len(x_names_pca)))
    print('PCA Fields: {0}'.format(x_names_pca))
    return df, x_names_pca


def remove_high_vif(X: pd.DataFrame, max_num: Union[int,float]=None):
    num_vars = X.shape[1]
    colnames = X.columns.values
    if not max_num or max_num == 0:
        max_num = round(np.power(X.shape[0], 0.3), 0)

    if num_vars > max_num:
        print('Removing variables with high VIF. New variable qty will be: [{0}]'.format(max_num))

        from joblib import Parallel, delayed
        while num_vars > max_num:
            vif = Parallel(n_jobs=-2, verbose=-1)(delayed(variance_inflation_factor)(X.loc[:, colnames].values, ix) for ix in range(X.loc[:, colnames].shape[1]))

            maxloc = vif.index(max(vif))
            print('dropping \'' + X.loc[:, colnames].columns[maxloc] + '\' at index: ' + str(maxloc))
            del colnames[maxloc]

        print('Remaining variables:')
        print(colnames)

    return X.loc[:, colnames]

def calc_equity_alloc() -> pd.Series:
    nonfin_biz_equity_liab = fred.get_series('NCBEILQ027S', observation_start=start_dt, observation_end=end_dt)
    nonfin_biz_credit_liab = fred.get_series('BCNSDODNS', observation_start=start_dt, observation_end=end_dt)
    household_nonprofit_credit_liab = fred.get_series('CMDEBT', observation_start=start_dt, observation_end=end_dt)
    fedgov_credit_liab = fred.get_series('FGSDODNS', observation_start=start_dt, observation_end=end_dt)
    localgov_ex_retirement_credit_liab = fred.get_series('SLGSDODNS', observation_start=start_dt, observation_end=end_dt)
    fin_biz_equity_liab = fred.get_series('FBCELLQ027S', observation_start=start_dt, observation_end=end_dt)
    restofworld_credit_liab = fred.get_series('DODFFSWCMI', observation_start=start_dt, observation_end=end_dt)

    # Divide nonfinancial and financial business equity reliabilities by all credit instrument liability in the economy
    equity_alloc = pd.Series(
        (
            (
                nonfin_biz_equity_liab
                + fin_biz_equity_liab
            )
            / 1000
        ) \
        / (
            (
                (
                    nonfin_biz_equity_liab
                    + fin_biz_equity_liab
                )
                / 1000
            )
            + (
                nonfin_biz_credit_liab
                + household_nonprofit_credit_liab
                + fedgov_credit_liab
                + localgov_ex_retirement_credit_liab
                + restofworld_credit_liab
            )
        )
    )
    return make_qtrly(equity_alloc, 'first')


def get_nyse_margin_debt(field_name: str) -> pd.Series:
    url = 'http://www.nyxdata.com/nysedata/asp/factbook/table_export_csv.asp?mode=tables&key=50'
    with requests.Session() as s:
        download = s.get(url=url)

    strio = io.StringIO(download.text)
    df = pd.read_table(strio, sep='\\t', skiprows=3)

    df['End of month'] = pd.DatetimeIndex(pd.to_datetime(df['End of month']),
                                  dtype=datetime.date).to_period('M').to_timestamp('M')
    df.set_index(['End of month'],
                 drop=True,
                 inplace=True,
                 verify_integrity=True)

    df = df\
        .replace( '[\$,)]', '', regex=True)\
        .replace( '[(]', '-',   regex=True)\
        .astype(float)

    # print(df)
    return make_qtrly(df[field_name], 'first')

#########################################################################################
# START THE STUFF
##########################################################################################
data_sources = dict()
data_sources['cpi_urb_nonvol'] = data_source('CPILFESL', 'fred')
data_sources['netexp_nom'] = data_source('NETEXP', 'fred')
data_sources['gdp_nom'] = data_source('GDP', 'fred')
data_sources['sp500'] = data_source('P', 'schiller')
data_sources['cape'] = data_source('CAPE', 'schiller')
data_sources['tsy_3mo_yield'] = data_source('DGS5', 'fred')
data_sources['tsy_5yr_yield'] = data_source('DGS5', 'fred')
data_sources['tsy_10yr_yield'] = data_source('DGS10', 'fred')
data_sources['tsy_30yr_yield'] = data_source('DGS30', 'fred')
data_sources['fed_funds_rate'] = data_source('FEDFUNDS', 'fred')
data_sources['gold_fix_3pm'] = data_source('GOLDPMGBD228NLBM', 'fred')
data_sources['unempl_rate'] = data_source('UNRATE', 'fred')
data_sources['industrial_prod'] = data_source('INDPRO', 'fred')
data_sources['empl_construction'] = data_source('USCONS', 'fred')
data_sources['equity_alloc'] = data_source(None, calc_equity_alloc)
data_sources['fed_reserves_tot'] = data_source('RESBALNS', 'fred')
data_sources['monetary_base_tot'] = data_source('BOGMBASE', 'fred')
data_sources['monetary_base_balances'] = data_source('BOGMBBMW', 'fred')
data_sources['fed_excess_reserves'] = data_source('EXCSRESNS', 'fred')
data_sources['sp500_peratio'] = data_source('MULTPL/SHILLER_PE_RATIO_MONTH', 'quandl')
data_sources['housing_starts'] = data_source('HOUST', 'fred')
data_sources['housing_supply'] = data_source('MSACSR', 'fred')
data_sources['real_estate_loans'] = data_source('REALLN', 'fred')
data_sources['med_house_price'] = data_source('MSPNHSUS', 'fred')
data_sources['med_family_income'] = data_source('MEFAINUSA646N', 'fred')
data_sources['combanks_business_loans'] = data_source('BUSLOANS', 'fred')
data_sources['combanks_assets_tot'] = data_source('TLAACBW027SBOG', 'fred')
data_sources['mortage_debt_individuals'] = data_source('MDOTHIOH', 'fred')
data_sources['capacity_util_tot'] = data_source('CAPUTLB50001SQ', 'fred')
data_sources['capacity_util_mfg'] = data_source('CUMFNS', 'fred')
data_sources['capacity_util_chem'] = data_source('CAPUTLG325S', 'fred')
data_sources['foreign_dir_invest'] = data_source('ROWFDNQ027S', 'fred')
data_sources['pers_savings_rt'] = data_source('PSAVERT', 'fred')
data_sources['gross_savings'] = data_source('GSAVE', 'fred')
data_sources['tax_receipts_corp'] = data_source('FCTAX', 'fred')
data_sources['tax_receipts_tot'] = data_source('W006RC1Q027SBEA', 'fred')
data_sources['nonfin_equity'] = data_source('MVEONWMVBSNNCB', 'fred')
data_sources['nonfin_networth'] = data_source('TNWMVBSNNCB', 'fred')
data_sources['nonfin_pretax_profit'] = data_source('NFCPATAX', 'fred')
data_sources['mzm_velocity'] = data_source('MZMV', 'fred')
data_sources['m2_velocity'] = data_source('M2V', 'fred')
data_sources['m1_velocity'] = data_source('M1V', 'fred')
data_sources['mzm_moneystock'] = data_source('MZMSL', 'fred')
data_sources['m2_moneystock'] = data_source('M2NS', 'fred')
data_sources['m1_moneystock'] = data_source('M1NS', 'fred')
# data_sources['wrk_age_pop_pct'] = data_source('SP.POP.1564.TO.ZS', 'worldbank', rerun=True)
data_sources['wrk_age_pop'] = data_source('LFWA64TTUSM647N', 'fred')
data_sources['employment_pop_ratio'] = data_source('EMRATIO', 'fred')
data_sources['nyse_margin_debt'] = data_source('Margin debt', get_nyse_margin_debt)
data_sources['nyse_margin_credit'] = data_source('Credit balances in margin accounts', get_nyse_margin_debt)

data_sources['recession_usa'] = data_source('USREC', 'fred')

ds_names = [k for k in data_sources.keys()]

raw_data_file = 'sp500_hist.p'
resave_data = False
try:
    if rawdata_from_file:
        with open(raw_data_file, 'rb') as f:
            (data_sources_temp,) = pickle.load(f)
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
    with open(raw_data_file, 'wb') as f:
        pickle.dump((data_sources,), f)

    # modified_z_score = 0.6745 * abs_dev / y_mad
    # modified_z_score[y == m] = 0
    # return modified_z_score > thresh


# from pandas_datareader import data, wb
# lol = data.DataReader(['TSLA'], 'yahoo', start_dt, end_dt)

final_data_file = 'sp500_final_data.p'
try:
    if finaldata_from_file:
        with open(final_data_file, 'rb') as f:
            df, x_names = pickle.load(f)
            print('Per settings, loaded data from file [{0}]'.format(final_data_file))
    else:
        raise ValueError('Per Settings, Reloading Data From Yahoo Finance/FRED/Everything Else.')
except Exception as e:
    print(e)
    df = pd.DataFrame()
    for k, ds in data_sources.items():
        if ds.provider in ('eod_hist', 'fred', 'schiller'):
            df[k] = make_qtrly(ds.data, 'first')
        else:
            df[k] = ds.data


    df['tsy_3mo_yield'] = df['tsy_3mo_yield'] / 100.
    df['tsy_5yr_yield'] = df['tsy_5yr_yield'] / 100.
    df['tsy_10yr_yield'] = df['tsy_10yr_yield'] / 100.
    df['tsy_30yr_yield'] = df['tsy_30yr_yield'] / 100.
    df['netexp_pct_of_gdp'] = (df['netexp_nom'] / df['gdp_nom'])
    df['base_minus_fed_res_tot'] = df['monetary_base_tot'] - df['fed_reserves_tot']
    df['med_family_income_vs_house_price'] = df['med_house_price'] / df['med_family_income']
    df['real_med_family_income'] =  df['med_family_income'] / df['cpi_urb_nonvol']
    df['tsy_10yr_minus_cpi'] = df['tsy_10yr_yield'] - df['cpi_urb_nonvol']
    df['tsy_10yr_minus_fed_funds_rate'] = df['tsy_10yr_yield'] - df['fed_funds_rate']
    df['tsy_3m10y_curve'] = df['tsy_3mo_yield'] / df['tsy_10yr_yield']
    df['tobin_q'] = [math.sqrt(x * y) for x, y in df.loc[:,['nonfin_equity','nonfin_networth']].values]  # geom mean
    df['corp_profit_margins'] = df['nonfin_pretax_profit'] / df['gdp_nom']
    df['mzm_usage'] = df['mzm_velocity'] * df['mzm_moneystock']
    df['m1_usage'] = df['m1_velocity'] * df['m1_moneystock']
    df['m2_usage'] = df['m2_velocity'] * df['m2_moneystock']
    df['nyse_margin_debt_ratio'] = df['nyse_margin_debt'] / df['nyse_margin_credit']

    x_names = [
        'equity_alloc',
        'tsy_10yr_yield', # Treasury prices have been generally increasing over the time period. Don't use.,
        'tsy_5yr_yield', # Treasury prices have been generally increasing over the time period. Don't use.,
        'tsy_3mo_yield', # Treasury prices have been generally increasing over the time period. Don't use.
        # , 'diff_tsy_10yr_and_cpi' # Makes the models go FUCKING CRAZY,
        'unempl_rate',
        # , 'empl_construction'  # Construction employees heave been generally increasing over the time period. Don't use.,
        # 'sp500_peratio',
        'capacity_util_mfg',
        'capacity_util_chem',
        # , 'gold_fix_3pm' # Gold price has been generally increasing over the time period. Don't use.
        # , 'fed_funds_rate' # Fed funds rate has been generally declining over the time period. Don't use.,
        'tsy_3m10y_curve',
        'industrial_prod',
        # , 'tsy_10yr_minus_fed_funds_rate'
        # , 'tsy_10yr_minus_cpi'
        # , 'netexp_pct_of_gdp' # Will cause infinite values when used with SHIFT (really any y/y compare)
        # , 'gdp_nom' # GDP is generally always rising. Don't use.
        # , 'netexp_nom' # Will cause infinite values when used with SHIFT (really any y/y compare)
        # , 'base_minus_fed_res_adj' # May also make the models go FUCKING CRAZY # Not much history
        # , 'tsy_30yr_yield' # Not much history,
        'med_family_income_vs_house_price',
        # , 'pers_savings_rt',
        'corp_profit_margins',
        'cape',
        'tobin_q',
        # , 'mzm_velocity'
        # , 'm2_velocity'
        # , 'm1_velocity',
        'mzm_usage',
        'm2_usage',
        'm1_usage',
        'employment_pop_ratio',
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

    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

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
            temp_df, new_x_names = get_diff_from_trend(df[x], x)

            # print(new_x_names)
            df = df.join(temp_df[new_x_names], how='inner')
            trend_x_names.extend(new_x_names)

    ##########################################################################################################
    # Derive difference variables to demonstrate changes from period to period
    ##########################################################################################################
    print('Adding x-year diff terms.')
    diff_x_names = [
        # 'gdp_nom',
        'equity_alloc',
        'cpi_urb_nonvol',
        # 'empl_construction',
        'industrial_prod',
        'housing_starts',
        'housing_supply',
        'med_house_price',
        'med_family_income',
        'unempl_rate',
        'industrial_prod',
        'tsy_10yr_yield',
        'tsy_5yr_yield',
        'tsy_3mo_yield',
        # 'tsy_10yr_minus_fed_funds_rate',
        # 'tsy_10yr_minus_cpi',
        'real_med_family_income',
        'combanks_business_loans',
        'combanks_assets_tot',
        'mortage_debt_individuals',
        'real_estate_loans',
        'foreign_dir_invest',
        # 'pers_savings_rt',
        # 'gross_savings',
        'tax_receipts_corp',
        'fed_funds_rate',
        # 'gold_fix_3pm',
        'corp_profit_margins',
        'cape',
        'tobin_q',
        # 'mzm_velocity',
        # 'm2_velocity',
        # 'm1_velocity',
        # 'mzm_moneystock',
        # 'm1_moneystock',
        # 'm2_moneystock',
        'mzm_usage',
        'm2_usage',
        'm1_usage',
        'employment_pop_ratio',
        'nyse_margin_debt',
        # 'nyse_margin_credit',  # has an odd discontunuity in the credit balances in Jan 85. Adjust before using.
        # 'nyse_margin_debt_ratio',  # has an odd discontunuity in the credit balances in Jan 85. Adjust before using.
    ]

    ##########################################################################################################
    # Interactions between each x value and its previous values
    ##########################################################################################################
    for name in diff_x_names:
        for y in [6]:
            diff_field_name = '{0}_{1}qtr_diff'.format(name, y)
            df[diff_field_name] = shift_x_quarters(df[name].pow(1./2.), y)
            # df[diff_field_name] = shift_x_quarters(df[name].apply(np.log, args=(1.5,)), y)
            x_names.append(diff_field_name)
    print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Value Imputation
    ##########################################################################################################
    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

    ##########################################################################################################
    # If even after imputation, some fields are empty, then you need to remove them
    ##########################################################################################################
    for n in x_names.copy():
        if df[n].isnull().any():
            print('Field [{0}] was still empty after imputation! Removing it!'.format(n))
            x_names.remove(n)

    ##########################################################################################################
    # Convert all x fields to EWMA versions, to smooth craziness
    ##########################################################################################################
    print('Converting fields to EWMA fields.')
    new_x_names = []
    for v in x_names:
        new_field_name = v + '_ewma'
        if new_field_name not in df.columns.values:
            df[new_field_name] = df[v].ewm(alpha=ewm_alpha).mean()
        new_x_names.append(new_field_name)
    x_names = new_x_names

    ##########################################################################################################
    # Trim all x fields to a threshold of 4 STDs
    ##########################################################################################################
    if trim_vars:
        print('Converting fields to trimmed fields.')
        new_x_names = []
        for v in x_names:
            new_field_name = v + '_trim'
            if new_field_name not in df.columns.values:
                df[new_field_name] = trim_outliers(df[v], thresh=4)
            new_x_names.append(new_field_name)
        x_names = new_x_names

    ##########################################################################################################
    # Value Imputation
    ##########################################################################################################
    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

    ##########################################################################################################
    # Create and add any interaction terms
    ##########################################################################################################
    corr_x_names = None
    if correlation_type == 'level_1':
        print('Creating correlation interaction terms [{0}]'.format(correlation_type))
        df, corr_x_names = get_level1_correlations(df=df, x_names=x_names)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Create interaction terms between the various x variables
    ##########################################################################################################
    print('Creating direct interaction terms [{0}]'.format(interaction_type))
    if interaction_type == 'all':
        new_x_names = []
        get_all_interactions(x_names)
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
            x_names = reduce_vars_corr(df=df, field_names=x_names, max_num=max_variables)
            print('X Names Length: {0}'.format(len(x_names)))
        elif dimension_method == 'pca':
            df, x_names = reduce_vars_pca(df=df, field_names=x_names, max_num=max_variables)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Create squared and squared-root versions of all the x fields
    ##########################################################################################################
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
        df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)
        print('X Names Length: {0}'.format(len(x_names)))

    if trend_x_names:
        for v in trend_x_names:
            new_field_name = v + '_ewma'
            if new_field_name not in df.columns.values:
                df[new_field_name] = df[v].ewm(alpha=ewm_alpha).mean()
            x_names.append(new_field_name)
        # IMPUTE VALUES!!!
        df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)
        print('X Names Length: {0}'.format(len(x_names)))

    ################################################################################################################
    # DIMENSION REDUCTION: Remove any highly correlated items from the regression, to reduce issues with the model #
    ################################################################################################################
    if (dimension_method is not None) and max_variables >= 0:
        if dimension_method == 'corr':
            x_names = reduce_vars_corr(df=df, field_names=x_names, max_num=max_variables)
            print('X Names Length: {0}'.format(len(x_names)))
        elif dimension_method == 'pca':
            df, x_names = reduce_vars_pca(df=df, field_names=x_names, max_num=max_variables)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Add x-variable for time since the last recession. Create y-variable for time until the next recession.
    ##########################################################################################################
    last_rec = -1
    for idx, period in enumerate(df.index.values):
        rec_val = df.get_value(period, 'recession_usa')
        if rec_val == 1:
            for i, v in enumerate(df.iloc[last_rec:idx, :].index.values):
                df.set_value(v, next_rec_field_name, idx - last_rec - i)
            last_rec = idx
            if idx > 0:
                next_val = min(df[prev_rec_field_name].iloc[idx-1] - 1, 0)
            else:
                next_val = -3
            df.set_value(period, prev_rec_field_name, next_val)
        elif rec_val == 0:
            df.set_value(period, prev_rec_field_name, idx - last_rec)
    x_names.append(prev_rec_field_name)

    # print(*df.index.values, sep='\n')

    if do_predict_next_recession:
        df[x_names+[next_rec_field_name]] = impute_if_any_nulls(df[x_names+[next_rec_field_name]], imputer=default_imputer)
    else:
        # x_names.append(next_rec_field_name)
        df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

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


    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

    ################################################################################################################
    # DIMENSION REDUCTION: Remove any highly correlated items from the regression, to reduce issues with the model #
    ################################################################################################################
    if (dimension_method is not None) and max_variables >= 0:
        if dimension_method == 'corr':
            x_names = reduce_vars_corr(df=df, field_names=x_names, max_num=max_variables)
            print('X Names Length: {0}'.format(len(x_names)))
        elif dimension_method == 'pca':
            df, x_names = reduce_vars_pca(df=df, field_names=x_names, max_num=max_variables)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # VARIANCE REDUCTION: Remove any highly correlated fields and/or use pca to eliminate correlation.
    ##########################################################################################################

    if (correlation_method is not None) and max_correlation < 1. and dimension_method != 'pca':
        if dimension_method == 'corr':
            x_names = reduce_variance_corr(df=df, fields=x_names, max_corr_val=max_correlation)
        elif correlation_method == 'pca':
            df, x_names = reduce_variance_pca(df=df, field_names=x_names, explained_variance=max_correlation)
        print('X Names Length: {0}'.format(len(x_names)))

for n in x_names:
    if df[n].isnull().any():
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
with open(final_data_file, 'wb') as f:
    pickle.dump((df, x_names), f)

# Predict the number of quarters until the next recession
if do_predict_next_recession:
    for d, report_name in predict_recession_time(df=df,
                                                 x_names=x_names,
                                                 y_field_name=next_rec_field_name,
                                                 model_set=next_recession_models):
        for k, v in d.items():
            df[k] = v

        # new_field_name = 'next_recession_{0}'.format(report_name)
        x_names.append([v for v in d.keys()][-1])

    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

# RECESSION PREDICTIONS
if do_predict_recessions:
    for yf in recession_predict_quarters_forward:
        for d, report_name in predict_recession(df=df,
                                                x_names=x_names,
                                                y_field_name=recession_field_name,
                                                quarters_forward=yf,
                                                model_set=recession_models):
            for k, v in d.items():
                df[k] = v

            new_field_name = 'recession_{0}'.format(report_name)
            for v in [v for v in list(d)][-2:]:
                x_names.append(v)

    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)

if do_predict_returns:
    for yf in returns_predict_quarters_forward:
        for d, report_name in predict_returns(df=df,
                                              x_names=x_names,
                                              y_field_name=sp_field_name,
                                              quarters_forward=yf,
                                              model_set=returns_models):
            for k, v in d.items():
                df[k] = v
            x_names.append([v for v in d.keys()][-1])

            new_field_name = 'sp500_{0}'.format(report_name)

    df[x_names] = impute_if_any_nulls(df[x_names], imputer=default_imputer)





##### OLD STUFF

# CREATE CORRELATION MATRIX AND

# for i in range(2, len(x_names)+1):
#     for x_name_combo in combinations(x_names, i):
#         # print("Evaluating Combinations: {0}".format(x_name_combo))
#         for operator_perm in possible_operator_permutations[len(x_name_combo)]:
#             for index, x_name in enumerate(x_name_combo):
#                 if index == 0:
#                     new_name = x_name
#                     new_series = pd.Series(df[x_name])
#                 else:
#                     # print("Correlation Coefficient: {0}".format(np.corrcoef(new_series, df[x_name])[0][1]))
#                     if abs(np.corrcoef(new_series, df[x_name])[0][1]) <= 0.3:
#                         # Interactions between two different fields, generated through division
#                         new_name = '{0}_{1}_{2}'.format(new_name, operator_perm[index-1], x_name)
#                         new_series = get_operator_fn(operator_perm[index-1])(new_series, df[x_name].replace({0: np.nan}))
#                     else:
#                         break
#             else:
#                 # print("Adding new interaction variable to DF: {0}".format(new_name))
#                 new_x_names.append(new_name)
#                 df[new_name] = new_series
# x_names.extend(new_x_names)