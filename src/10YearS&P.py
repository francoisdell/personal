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
import bls
import operator
# from matplotlib import rcParams
# import yahoo_finance
from joblib import Parallel, delayed
import itertools
from statsmodels import robust
from fredapi import Fred
from matplotlib import pyplot as plt
from matplotlib.ticker import OldScalarFormatter
from sklearn.decomposition import TruncatedSVD
from inspect import isfunction
from pandas.compat import StringIO
import requests_cache
import Model_Builder as mb
from collections import OrderedDict
from typing import Union, Callable
import io
import requests
import importlib

# Specify whether the data will be loaded from pickle files or recollected fresh from the interwebz
rawdata_from_file = True  # Whether to load the raw data (pre-transformations) from a pickle file
finaldata_from_file = False  # Whether to load the final data (post-transformations) from a pickle file

# Do you want to predict returns? Or Recessions?
do_predict_returns = False
returns_predict_years_forward = [9, 10]
do_predict_recessions = True
recession_predict_years_forward = [2]

# If you want to remove variables that don't meet a certain significance level, set this < 1. Requiring 95% = 5.0e-2.
selection_limit = 5.0e-2

interaction_type = 'level_1'  # Specify whether to derive pairwise interaction variables. Options: all, level_1, None
correlation_type = 'level_1'  # Specify whether to derive pairwise EWM-correlation variables. Options: level_1, None

# DIMENSION REDUCTION. Recommend you use either PCA or a maximum value for variance. Otherwise the data could get big.
pca_variance = None  # Options: None (if you don't want PCA) or a float 0-1 for the amount of explained variance desired.
max_var_correlation = 0.99  # Options: None (if you don't want to remove vars) or a float 0-1. 0.8-0.9 is usually good.

# Variables specifying what kinds of predictions to run, and for what time period
start_dt = '1920-01-01'
end_dt = datetime.today().strftime('%Y-%m-%d')
train_pct = 0.8  # Defines the train/test split
real = False
sp_field_name = 'sp500'
recession_field_name = 'recession_usa'
verbose = False
fred = Fred(api_key='b604ef6dcf19c48acc16461e91070c43')
ewm_halflife = 4.2655*2  # Halflife for EWM calculations. 4.2655 corresponds to a 0.125 weight.
default_imputer = 'knnimpute'  # 'fancyimpute' or 'knnimpute'. knnimpute is generally much faster, if less ideal.

recession_models = [
                   # mb.ModelSet(final_models=['logit','pass_agg_c','nearest_centroid'],
                   #              initial_models=['logit','pass_agg_c','nearest_centroid','gbc'])
                    # 'gauss_proc_c'
                    mb.ModelSet(final_models=['logit','nearest_centroid','etree_c','pass_agg_c','knn_c','gbc','svc','sgd_c','bernoulli_nb','ridge_c'],
                                initial_models=['logit','nearest_centroid','etree_c','pass_agg_c','knn_c','gbc','svc','bernoulli_nb'])
                  # ,['sgd_c','svc','knn_c','bernoulli_nb','nearest_centroid','gbc','logit','rfor','etree_c','pass_agg_c']  # 2yr:   ||  3yr:
                  # ,['sgd_c','knn_c','bernoulli_nb','rfor','gbc','pass_agg_c','logit','svc','etree_c','nearest_centroid']  # 2yr:   ||  3yr:
                  # ,['sgd_c','knn_c','gbc','pass_agg_c','rfor','logit','svc','nearest_centroid','etree_c','bernoulli_nb']  # 2yr:   ||  3yr:
                  # ,['sgd_c','gbc','pass_agg_c','logit','svc','rfor','nearest_centroid','bernoulli_nb','etree_c','knn_c']  # 2yr:   ||  3yr:
                  # ,['sgd_c','svc','logit','rfor','knn_c','bernoulli_nb','nearest_centroid','gbc','pass_agg_c','etree_c']  # 2yr:   ||  3yr:
                  # ,['sgd_c','svc','logit','knn_c','bernoulli_nb','nearest_centroid','pass_agg_c','rfor','etree_c','gbc']  # 2yr:   ||  3yr:
                  ]

## INTERESTING @ 2 YEARS
# recession_models = [
#                   ['abc','neural_c','knn_c','sgd_c','bernoulli_nb','nearest_centroid','pass_agg_c','gbc']
#                   ]

# recession_models = ['gbc',
#                   'abc',
#                   'neural_c',
#                   'knn_c',
#                   'sgd_c',
#                   'pass_agg_c',
#                   'bernoulli_nb',
#                   'nearest_centroid',
#                   'ridge_c',
#                   ['gbc','abc','knn_c','sgd_c','pass_agg_c','bernoulli_nb','nearest_centroid','neural_c']
#                   ]

returns_models = [
                # 'knn_r',
                # 'elastic_net_stacking',
                # 'gbr',
                # 'neural_r',
                # 'ridge'
                 ['rfor_r','svr','sgd_r','gbr','neural_r','pass_agg_r','elastic_net','knn_r','ridge']  # 5yr: 0
                ,['rfor_r','sgd_r','gbr','neural_r','pass_agg_r','elastic_net','knn_r','ridge','svr']  # 5yr: 0
                ,['rfor_r','svr','gbr','neural_r','pass_agg_r','elastic_net','knn_r','ridge','sgd_r']  # 5yr: 0
                ,['rfor_r','svr','sgd_r','gbr','neural_r','elastic_net','knn_r','ridge','pass_agg_r']  # 5yr: 0
                ,['rfor_r','svr','sgd_r','gbr','neural_r','pass_agg_r','knn_r','ridge','elastic_net']  # 5yr: 0
                ,['rfor_r','svr','sgd_r','gbr','pass_agg_r','knn_r','ridge','elastic_net','neural_r']  # 5yr: 0
                ]

if real:
    sp_field_name += '_real'

#use the list to filter the local namespace
# safe_func_list = ['first', 'last', 'mean']
# safe_funcs = dict([(k, locals().get(k, None)) for k in safe_func_list])

def make_qtrly(s: pd.Series, t: str='mean') -> pd.Series:
    s.index = pd.DatetimeIndex(s.index.values)
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
        fig.savefig(save_name)
    print("Showing Plot...")
    plt.show()


def get_obj_name(o) -> str:
    return [k for k, v in locals().items() if v is o][0]

def reverse_enumerate(l):
   for index in reversed(range(len(l))):
      yield index, l[index]

def predict_returns(df: pd.DataFrame, x_names: list, y_field_name: str, model_name: Union[str, list]
                    , years_forward: int, prune: bool=False) -> (OrderedDict, str):

    print("\n-----------------------"
          "\n   YEARS FORWARD: {0} "
          "\n-----------------------".format(years_forward))

    if years_forward > 0:
        forward_y_field_name = '{0}_{1}yr'.format(y_field_name, years_forward)
        df[forward_y_field_name] = 1 - pd.Series(df[y_field_name].shift(years_forward * 4) / df[y_field_name]).pow(
            1 / float(years_forward)).shift(-years_forward * 4)

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
    if isinstance(model_name, str):
        models = [model_name]
    report_name = 'returns_{0}yr_{1}'.format(years_forward, model_name[0] if len(model_name) == 1 else 'stacked')
    df = mb.predict(df
                    , x_fields=x_fields
                    , y_field=y_field
                    , model_type=model_name
                    , report_name=report_name
                    , show_model_tests=True
                    , retrain_model=True
                    , selection_limit=selection_limit
                    , predict_all=True
                    , verbose=verbose
                    , train_pct=train_pct
                    , random_train_test=False)

    forward_y_field_name_pred = 'pred_' + forward_y_field_name
    #################################

    invest_amt_per_qtr = 5000
    df['invest_strat_cash'] = invest_amt_per_qtr
    df['invest_strat_basic'] = invest_amt_per_qtr
    df['invest_strat_mixed'] = invest_amt_per_qtr
    df['invest_strat_equity'] = invest_amt_per_qtr
    df['invest_strat_tsy'] = invest_amt_per_qtr

    mean_y_field_return = df[forward_y_field_name].mean()

    for i in range(df.shape[0] - years_forward * 4):
        df.ix[i + (4 * years_forward), 'invest_strat_cash'] += df.ix[i, 'invest_strat_cash']

        if mean_y_field_return >= df.ix[i, 'tsy_10yr_yield']:
            df.ix[i + (4 * years_forward), 'invest_strat_basic'] += df.ix[i, 'invest_strat_basic'] * (1 + df.ix[
                i, forward_y_field_name]) ** years_forward
        else:
            df.ix[i + (4 * years_forward), 'invest_strat_basic'] += df.ix[i, 'invest_strat_basic'] * (1 + df.ix[
                i, 'tsy_10yr_yield']) ** years_forward

        if df.ix[i, forward_y_field_name_pred] >= df.ix[i, 'tsy_10yr_yield']:
            df.ix[i + (4 * years_forward), 'invest_strat_mixed'] += df.ix[i, 'invest_strat_mixed'] * (1 + df.ix[
                i, forward_y_field_name]) ** years_forward
        else:
            df.ix[i + (4 * years_forward), 'invest_strat_mixed'] += df.ix[i, 'invest_strat_mixed'] * (1 + df.ix[
                i, 'tsy_10yr_yield']) ** years_forward

        df.ix[i + (4 * years_forward), 'invest_strat_equity'] += df.ix[i, 'invest_strat_equity'] * (1 + df.ix[
            i, forward_y_field_name]) ** years_forward

        df.ix[i + (4 * years_forward), 'invest_strat_tsy'] += df.ix[i, 'invest_strat_tsy'] * (1 + df.ix[
            i, 'tsy_10yr_yield']) ** years_forward

    df['total_strat_cash'] = df['invest_strat_cash'][train_mask].rolling(window=(years_forward * 4)).sum()
    df['total_strat_basic'] = df['invest_strat_basic'][train_mask].rolling(window=(years_forward * 4)).sum()
    df['total_strat_mixed'] = df['invest_strat_mixed'][train_mask].rolling(window=(years_forward * 4)).sum()
    df['total_strat_equity'] = df['invest_strat_equity'][train_mask].rolling(window=(years_forward * 4)).sum()
    df['total_strat_tsy'] = df['invest_strat_tsy'][train_mask].rolling(window=(years_forward * 4)).sum()

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
    df[forward_y_field_name_pred_fut] = df[forward_y_field_name_pred].astype(np.float64)
    df[forward_y_field_name_pred_fut][train_mask] = np.nan

    # print('\n===== Head =====\n', df.head(5))
    # print('\n===== Tail =====\n', df.tail(5))
    #
    # if hasattr(rm, 'intercept_'):
    #     print("Intercept:", rm.intercept_)
    #
    # print('==== Final Variables ====\n', np.asarray(x_names, dtype=np.str))
    #
    # if hasattr(rm, 'coef_'):
    #     print("==== Coefficients ====\n", rm.coef_)
    #
    # print('==== Score ====\n', rm.score(df.loc[train_mask, x_names].values.reshape(-1, len(x_names))
    #                                     , df.loc[train_mask, [y_field_name]].values))
    # scores, p_vals = f_regression(df.loc[train_mask, x_names].values.reshape(-1, len(x_names))
    #                               , df.loc[train_mask, [y_field_name]].values)
    # print('==== Scores ====\n', scores)
    # print('==== P-Values ====\n', p_vals)

    print(max(df.index))
    from dateutil.relativedelta import relativedelta
    df = df.reindex(
        pd.DatetimeIndex(start=df.index.min(), end=max(df.index) + relativedelta(years=years_forward), freq='1Q'))

    y_field_name_pred = '{0}_pred'.format(y_field_name)
    df[y_field_name_pred] = (df[y_field_name] * df[forward_y_field_name_pred].add(1).pow(years_forward)).shift(years_forward * 4)

    # if years_forward in [10]:
    chart(df
          , ys=[[forward_y_field_name, forward_y_field_name_pred, 'tsy_10yr_yield'], [y_field_name, y_field_name_pred]]
          , invert=[False, False]
          , log_scale=[False, True]
          , save_name='_'.join([forward_y_field_name_pred, report_name, str(years_forward)])
          , title='_'.join([y_field_name, report_name]))

    return (OrderedDict(((forward_y_field_name, df[forward_y_field_name])
        , (forward_y_field_name_pred, df[forward_y_field_name_pred])
        , (y_field_name_pred, df[y_field_name_pred])))
        , report_name)



def predict_recession(df: pd.DataFrame
                      , x_names: list
                      , y_field_name: str
                      , years_forward: int
                      , model_set: mb.ModelSet)\
        -> (OrderedDict, str):

    print("\n-----------------------"
          "\n   YEARS FORWARD: {0} "
          "\n-----------------------".format(years_forward))

    if years_forward > 0:
        forward_y_field_name = '{0}_{1}yr'.format(y_field_name, years_forward)
        df[forward_y_field_name] = df[y_field_name].shift(-years_forward * 4)

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

    report_name = 'recession_{0}yr_{1}'.format(years_forward, model_set)

    for df, final_model_name in mb.predict(df
                    , x_fields=x_fields
                    , y_field=y_field
                    , model_type=model_set
                    , report_name=report_name
                    , show_model_tests=True
                    , retrain_model=False
                    , selection_limit=selection_limit
                    , predict_all=True
                    , verbose=verbose
                    , train_pct=train_pct
                    , random_train_test=False
                    ,pca_explained_var=1.0):

        forward_y_field_name_pred = 'pred_' + forward_y_field_name
        #################################

        print(max(df.index))
        from dateutil.relativedelta import relativedelta
        df = df.reindex(
            pd.DatetimeIndex(start=df.index.min(), end=max(df.index) + relativedelta(years=years_forward), freq='1Q'))

        y_field_name_pred = '{0}_pred'.format(y_field_name)
        df[y_field_name_pred] = df[forward_y_field_name_pred].shift(years_forward * 4)
        chart_y_field_names = [y_field_name, y_field_name_pred]

        y_pred_names = mb.get_pred_field_names()
        y_prob_field_name = '{0}_prob_1.0'.format(forward_y_field_name)
        if y_prob_field_name in y_pred_names:
            df[y_prob_field_name] = df[y_prob_field_name].shift(years_forward * 4)
            chart_y_field_names += [y_prob_field_name]

        report_name = 'recession_{0}yr_{1}_{2}'.format(years_forward, model_set, final_model_name)
        # if years_forward in [10]:
        chart(df
              , ys=[['sp500'], chart_y_field_names]
              , invert=[False, False]
              , log_scale=[True, False]
              , save_name='_'.join([forward_y_field_name_pred, report_name])
              , title='_'.join([y_field_name, report_name]))

        yield (OrderedDict(((forward_y_field_name, df[forward_y_field_name])
            , (forward_y_field_name_pred, df[forward_y_field_name_pred])
            , (y_field_name_pred, df[y_field_name_pred])))
            , report_name)

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

        if provider in ('fred', 'yahoo', 'quandl', 'schiller', 'eod_hist', 'bls'):
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
            self.data = self.provider()
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
        print("Collected data for [{0}]".format(self.code))


def shift_x_years(s: pd.Series, y: int):
    return s / s.shift(4*y)


def impute_if_any_nulls(df, imputer=default_imputer):
    if df.isnull().any().any():
        print('Running imputation')
        try:
            if imputer == 'knnimpute':
                raise ValueError('knnimpute requested')
            imputer = importlib.import_module("fancyimpute")
            solver = imputer.MICE(init_fill_method='random') # mean, median, or random
            # solver = fancyimpute.NuclearNormMinimization()
            # solver = fancyimpute.MatrixFactorization()
            # solver = fancyimpute.IterativeSVD()
            df.loc[:, x_names] = solver.complete(df.loc[:, x_names].values)
        except (ImportError, ValueError) as e:
            imputer = importlib.import_module("knnimpute")
            df.loc[:, x_names] = imputer.knn_impute_with_argpartition(df.loc[:, x_names].values,
                        missing_mask=np.isnan(df.loc[:, x_names].values), k=5)
        # df = solver.complete(df.values)
    return df



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
                # df[interaction_field_name] = df[v].ewm(halflife=ewm_halflife).corr(other=df[v1])
                # new_x_names.append(interaction_field_name)

    return df, new_x_names



def get_diff_std_and_flags(df: pd.DataFrame
                           , field_name: str
                           , groupby_fields: list=lambda x: True
                           , halflife: float=ewm_halflife
                           , stdev_qty: int=2)\
        -> (pd.DataFrame, list):

    ewma_field_name = field_name + '_ewma'
    if ewma_field_name not in df.columns:
        df[ewma_field_name] = df[field_name].ewm(halflife=ewm_halflife).mean()

    new_name_prefix = field_name + '_val_to_ewma'
    new_name_diff = new_name_prefix + '_diff'
    new_name_diff_std = new_name_prefix + '_diff_std'
    new_name_add_std = new_name_prefix + '_add_std'
    new_name_sub_std = new_name_prefix + '_sub_std'
    new_name_trend_fall = new_name_prefix + '_trend_fall_flag'
    new_name_trend_rise = new_name_prefix + '_trend_rise_flag'

    df[new_name_diff] = df[field_name] - df[ewma_field_name]
    df[new_name_diff_std] = df.groupby(by=[groupby_fields])[new_name_diff].apply(lambda x: x.ewm(halflife=halflife).std(bias=False))
    df[new_name_add_std] = df[ewma_field_name] + (stdev_qty * df[new_name_diff_std])
    df[new_name_sub_std] = df[ewma_field_name] - (stdev_qty * df[new_name_diff_std])
    df[new_name_sub_std] = df[new_name_sub_std].clip(lower=0)
    df[new_name_trend_rise] = df[field_name].values < df[new_name_sub_std].values
    df[new_name_trend_fall] = df[field_name].values > df[new_name_add_std].values

    return df, [new_name_diff, new_name_diff_std, new_name_add_std, new_name_sub_std
        , new_name_trend_rise, new_name_trend_fall]


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
                # s = pd.ewmcorr(df[v], df[v1], halflife=ewm_halflife)
                s = df[v].ewm(halflife=ewm_halflife).corr(other=df[v1])
                # s = df[v].apply(lambda x: x.fillna(0).ewm(halflife=ewm_halflife).corr(other=df[v1]))
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


def remove_correlated(df: pd.DataFrame, x_fields: list, max_corr_val: float):
    '''
    Obj: Drops features that are strongly correlated to other features.
          This lowers model complexity, and aids in generalizing the model.
    Inputs:
          df: features df (x)
          corr_val: Columns are dropped relative to the corr_val input (e.g. 0.8)
    Output: df that only includes uncorrelated features
    '''

    print('Removing one variable for each pair of variables with correlation greater than [{0}]'.format(max_corr_val))
    # Creates Correlation Matrix and Instantiates
    corr_matrix = df.loc[:, x_fields].corr()
    # iters = range(len(corr_matrix.columns) - 1)
    drop_cols = set()

    # Iterates through Correlation Matrix Table to find correlated columns
    # for i in iters:
    for i, v in enumerate(x_fields[:-1]):
        for j in reversed(range(i)):
            max_corr = corr_matrix.iloc[j:(j+1), (i+1):].max()
            if max_corr >= max_corr_val:
                # Prints the correlated feature set and the corr val
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.add(v)
                break
            # item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            # col = item.columns
            # row = item.index
            # val = item.values
            # if val >= max_corr_val:
            #     # Prints the correlated feature set and the corr val
            #     # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            #     drop_cols.add(v)
            #     break

    return_x_vals = [v for v in x_fields if v not in list(drop_cols)]
    return return_x_vals
    # drops = sorted(set(drop_cols))[::-1]

    # Drops the correlated columns
    # for i in drops:
    #     col = x.iloc[:, (i+1):(i+2)].columns.values
    #     df = x.drop(col, axis=1)

    # return df


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
    return make_qtrly(equity_alloc, 'last')


def convert_to_pca(pca_df: pd.DataFrame, field_names: list, explained_variance: float=0.95):
    print("Conducting PCA and pruning components above the desired explained variance ratio")
    max_components = len(field_names) - 1
    pca_model = TruncatedSVD(n_components=max_components, random_state=555)

    x_results = pca_model.fit_transform(pca_df.loc[:, field_names]).T
    # print(pca_model.components_)
    print('PCA explained variance ratios.')
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
    return x_names_pca


def get_nyse_margin_debt() -> pd.Series:
    url = 'http://www.nyxdata.com/nysedata/asp/factbook/table_export_csv.asp?mode=tables&key=50'
    with requests.Session() as s:
        download = s.get(url=url)

    strio = io.StringIO(download.text)
    df = pd.read_table(strio, sep='\\t', skiprows=3)

    print(df)

    df['End of month'] = pd.DatetimeIndex(pd.to_datetime(df['End of month'])).to_period('M').to_timestamp('M')
    df.set_index(['End of month'], inplace=True, drop=True)
    print(df)


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
data_sources['equity_alloc'] = data_source(None, globals()['calc_equity_alloc'])
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
            df[k] = make_qtrly(ds.data, 'last')
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



    if do_predict_returns:
        # FULL LIST FOR LINEAR REGRESSION
        x_names = [
            'equity_alloc'
            , 'tsy_10yr_yield'
            , 'tsy_5yr_yield'
            , 'tsy_3mo_yield'
            , 'cape'
            , 'tobin_q'
            # , 'diff_tsy_10yr_and_cpi' # Makes the models go FUCKING CRAZY
            , 'unempl_rate'
            , 'empl_construction'
            , 'sp500_peratio'
            , 'capacity_util_mfg'
            , 'capacity_util_chem'
            # , 'gold_fix_3pm'
            # , 'fed_funds_rate'
            , 'tsy_3m10y_curve'
            , 'industrial_prod'
            , 'tsy_10yr_minus_fed_funds_rate'
            , 'tsy_10yr_minus_cpi'
            # , 'netexp_pct_of_gdp' # Will cause infinite values when used with SHIFT (really any y/y compare)
            # , 'gdp_nom'
            # , 'netexp_nom' # Will cause infinite values when used with SHIFT (really any y/y compare)
            # , 'base_minus_fed_res_adj' # May also make the models go FUCKING CRAZY # Not much history
            # , 'tsy_30yr_yield' # Not much history
            , 'med_family_income_vs_house_price'
            , 'pers_savings_rt'
            , 'corp_profit_margins'

            ]
        """
        x_names = [
            'equity_alloc'
            # , 'tsy_10yr_yield'
            # , 'tsy_5yr_yield'
            # , 'tsy_3mo_yield'
            , 'cape'
            , 'tobin_q'
            # , 'diff_tsy_10yr_and_cpi' # Makes the models go FUCKING CRAZY
            , 'unempl_rate'
            # , 'empl_construction'
            , 'sp500_peratio'
            # , 'capacity_util_mfg'
            # , 'capacity_util_chem'
            # , 'gold_fix_3pm'
            # , 'fed_funds_rate'
            # , 'tsy_3m10y_curve'
            , 'industrial_prod'
            # , 'tsy_10yr_minus_fed_funds_rate'
            # , 'tsy_10yr_minus_cpi'
            # , 'netexp_pct_of_gdp' # Will cause infinite values when used with SHIFT (really any y/y compare)
            # , 'gdp_nom'
            # , 'netexp_nom' # Will cause infinite values when used with SHIFT (really any y/y compare)
            # , 'base_minus_fed_res_adj' # May also make the models go FUCKING CRAZY # Not much history
            # , 'tsy_30yr_yield' # Not much history
            , 'med_family_income_vs_house_price'
            # , 'pers_savings_rt'
            # , 'corp_profit_margins'
            ]
        """

    else:
        x_names = [
            'equity_alloc'
            # , 'tsy_10yr_yield' # Treasury prices have been generally increasing over the time period. Don't use.
            # , 'tsy_5yr_yield' # Treasury prices have been generally increasing over the time period. Don't use.
            # , 'tsy_3mo_yield' # Treasury prices have been generally increasing over the time period. Don't use.
            , 'cape'
            , 'tobin_q'
            # , 'diff_tsy_10yr_and_cpi' # Makes the models go FUCKING CRAZY
            , 'unempl_rate'
            # , 'empl_construction'  # Construction employees heave been generally increasing over the time period. Don't use.
            , 'sp500_peratio'
            , 'capacity_util_mfg'
            , 'capacity_util_chem'
            # , 'gold_fix_3pm' # Gold price has been generally increasing over the time period. Don't use.
            # , 'fed_funds_rate' # Fed funds rate has been generally declining over the time period. Don't use.
            , 'tsy_3m10y_curve'
            , 'industrial_prod'
            , 'tsy_10yr_minus_fed_funds_rate'
            # , 'tsy_10yr_minus_cpi'
            # , 'netexp_pct_of_gdp' # Will cause infinite values when used with SHIFT (really any y/y compare)
            # , 'gdp_nom' # GDP is generally always rising. Don't use.
            # , 'netexp_nom' # Will cause infinite values when used with SHIFT (really any y/y compare)
            # , 'base_minus_fed_res_adj' # May also make the models go FUCKING CRAZY # Not much history
            # , 'tsy_30yr_yield' # Not much history
            , 'med_family_income_vs_house_price'
            , 'pers_savings_rt'
            , 'corp_profit_margins'
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


    print('Adding x-year diff terms.')
    diff_x_names = [
        'gdp_nom'
        , 'cape'
        , 'tobin_q'
        , 'cpi_urb_nonvol'
        , 'empl_construction'
        , 'industrial_prod'
        , 'housing_starts'
        , 'housing_supply'
        , 'med_house_price'
        , 'med_family_income'
        , 'unempl_rate'
        , 'industrial_prod'
        , 'tsy_10yr_minus_fed_funds_rate'
        , 'tsy_10yr_minus_cpi'
        , 'real_med_family_income'
        , 'combanks_business_loans'
        , 'combanks_assets_tot'
        , 'mortage_debt_individuals'
        , 'real_estate_loans'
        , 'foreign_dir_invest'
        , 'pers_savings_rt'
        , 'gross_savings'
        , 'tax_receipts_corp'
        , 'fed_funds_rate'
        , 'gold_fix_3pm'
        , 'corp_profit_margins'
    ]

    ##########################################################################################################
    # Interactions between each x value and its previous values
    ##########################################################################################################
    for name in diff_x_names:
        for y in [1]:
            diff_field_name = '{0}_{1}yr_diff'.format(name, y)
            df[diff_field_name] = shift_x_years(df[name], y)
            x_names.append(diff_field_name)
    print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Value Imputation
    ##########################################################################################################
    df.loc[:, x_names] = impute_if_any_nulls(df.loc[:, x_names], imputer=default_imputer)

    ##########################################################################################################
    # Remove any highly correlated items from the regression, to reduce issues with the model
    ##########################################################################################################
    if max_var_correlation and max_var_correlation < 1:
        x_names = remove_correlated(df, x_fields=x_names, max_corr_val=max_var_correlation)
        print('X Names Length: {0}'.format(len(x_names)))

    # Perform the addition of the interaction terms
    corr_x_names = None
    print('Creating correlation interaction terms [{0}]'.format(correlation_type))
    if correlation_type == 'level_1':
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

    ##########################################################################################################
    # Remove any highly correlated items from the regression, to reduce issues with the model
    ##########################################################################################################
    if max_var_correlation and max_var_correlation < 1:
        x_names = remove_correlated(df, x_fields=x_names, max_corr_val=max_var_correlation)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Convert all x fields to EWMA versions, to smooth craziness
    ##########################################################################################################
    print('Converting fields to EWMA fields.')
    new_x_names = []
    for v in x_names:
        new_field_name = v + '_ewma'
        if new_field_name not in df.columns.values:
            df[new_field_name] = df[v].ewm(halflife=ewm_halflife).mean()
        new_x_names.append(new_field_name)
    x_names = new_x_names

    ##########################################################################################################
    # Trim all x fields to a threshold of 4 STDs
    ##########################################################################################################
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
    df.loc[:, x_names] = impute_if_any_nulls(df.loc[:, x_names], imputer=default_imputer)

    ##########################################################################################################
    # Create squared and squared-root versions of all the x fields
    ##########################################################################################################
    # Generate all possible combinations of the imput variables.
    print('Creating squared and square root varieties of predictor variables')
    operations = [('pow2', math.pow, (2,)), ('sqrt', math.sqrt, None)]
    new_x_names = []
    for v in x_names:
        for suffix, op, var in operations:
            new_x_name = '{0}_{1}'.format(v, suffix)
            df[new_x_name] = df[v].abs().apply(op, args=var) * df[v].apply(lambda x: -1 if x < 0 else 1)
            new_x_names.append(new_x_name)
    x_names.extend(new_x_names)
    print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Add the x value correlations (generated earlier) to the dataset
    ##########################################################################################################
    if corr_x_names:
        x_names.extend(corr_x_names)
        # IMPUTE VALUES!!!
        df.loc[:, x_names] = impute_if_any_nulls(df.loc[:, x_names], imputer=default_imputer)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # Derive special predictor variables
    ##########################################################################################################
    print('Deriving special predictor variables.')
    df, new_x_names = get_diff_std_and_flags(df, 'sp500')
    # x_names.extend(new_x_names)
    sp500_qtr_since_last_corr = 'sp500_qtr_since_last_corr'
    df[sp500_qtr_since_last_corr] = time_since_last_true(df[new_x_names[-1]])
    x_names.append(sp500_qtr_since_last_corr)

    ##########################################################################################################
    # Finally, remove any highly correlated items from the regression, to reduce issues with the model
    ##########################################################################################################
    if max_var_correlation and max_var_correlation < 1:
        x_names = remove_correlated(df, x_fields=x_names, max_corr_val=max_var_correlation)
        print('X Names Length: {0}'.format(len(x_names)))

    ##########################################################################################################
    # If desired, use PCA to reduce the predictor variables
    ##########################################################################################################
    if pca_variance and pca_variance < 1:
        x_names = convert_to_pca(df, x_names, explained_variance=pca_variance)
        print('X Names Length: {0}'.format(len(x_names)))

# print('===== Head =====\n', df.head(5))
# print('===== Tail =====\n', df.tail(5))

# print(df.head(5))
# print(df.tail(5))

# df_valid = df.loc[~train_mask, :]
# predict_years_forward = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

print('Dumping final dataset (post-transformations) to pickle file [{0}]'.format(final_data_file))
with open(final_data_file, 'wb') as f:
    pickle.dump((df, x_names), f)

if do_predict_returns:
    for yf in returns_predict_years_forward:
        for idx, model_list in enumerate(returns_models):
            d, report_name = predict_returns(df=df, x_names=x_names
                                , y_field_name=sp_field_name
                                , years_forward=yf
                                , model_name=model_list
                                , prune=True)
            for k,v in d.items():
                df[k] = v

            new_dataset = [v for v in d.values()][-1]
            new_field_name = 'sp500_{0}_m{1}'.format(report_name, idx)

# RECESSION PREDICTIONS
if do_predict_recessions:
    for yf in recession_predict_years_forward:
        for idx, model_set in enumerate(recession_models):
            for d, report_name in predict_recession(df=df
                                  , x_names=x_names
                                  , y_field_name=recession_field_name
                                  , years_forward=yf
                                  , model_set=model_set):
                for k,v in d.items():
                    df[k] = v

                new_dataset = [v for v in d.values()][-1]
                new_field_name = '{0}_m{1}'.format(report_name, idx)







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