import matplotlib
matplotlib.use('TkAgg')
import calendar
import math
import pickle
import os
from datetime import datetime as dt, timedelta as td
import numpy as np
import pandas as pd
import quandl
import wbdata
import bls
import operator
import itertools
from Function_Toolbox import impute_if_any_nulls, load_env
from sklearn import feature_selection as sk_feat_sel
from scipy.stats import logistic
from sklearn import preprocessing as sk_prep
from statsmodels import robust
from fredapi import Fred
from matplotlib import pyplot as plt
from matplotlib.ticker import OldScalarFormatter
from sklearn.decomposition import TruncatedSVD
from inspect import isfunction
from pandas.compat import StringIO
import requests_cache
from typing import Union, Callable
import io
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests


class WTFException(Exception):
    pass


def make_qtrly(s: pd.Series, t: str = 'first', name: str=None) -> pd.Series:
    s.index = pd.DatetimeIndex(s.index.values, dtype=dt.date)
    s.index.freq = s.index.inferred_freq
    name = name or s.name or ''
    # print(s)

    if t == 'mean':
        s = s.resample('1Q').mean().astype(np.float64)
    elif t == 'first':
        s = s.resample('1Q').first().astype(np.float64)
    elif t == 'last':
        s = s.resample('1Q').last().astype(np.float64)

    if s.isnull().any():
        print(f'Series {name} still has some empty data. Filling that in with the last known value.')
        s.fillna(method='ffill', inplace=True)

    # Conform everything to the end of the quarter
    idx = s.index
    for i, v in enumerate(idx):
        v.replace(month=math.ceil(v.month / 3) * 3)
        v.replace(day=calendar.monthrange(v.year, v.month)[-1])
    s.index = idx

    # s.index = s.index + pd.Timedelta(3, unit='M') - pd.Timedelta(1, unit='d')

    # s.index = pd.to_datetime([d + relativedelta(days=1) for d in s.index])
    # s.index.freq = s.index.inferred_freq

    # I wanted to make this function more dynamic and eliminate the if/else bullshit, with the below line (which failed)
    # s = s.resample('3MS').apply(eval(t + '(self)', {"__builtins__": None}, safe_funcs)).astype(np.float64)

    # print(s)
    return s


def get_closes(d: list, fld_names: list = list(('Close', 'col3'))) -> pd.Series:
    index_list = []
    val_list = []
    date_names = ['Date', 'col0']
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


# df = Pandas Dataframe
# ys = [ [cols in the same y], [cols in the same y], [cols in the same y], .. ]
# invert = [[True/False to invert y axis], [True/False to invert y axis], [True/False to invert y axis], ..]
def chart(df: pd.DataFrame, ys: list, invert: list, log_scale: list, save_addr: str=None, title: str=None,
          show: bool=True):
    from itertools import cycle
    fig, ax = plt.subplots()

    axes = [ax]
    for y in ys[1:]:
        # Twin the x-axis twice to make independent y-axes.
        axes.append(ax.twinx())

    extra_ys = len(axes[2:])

    # Make some space on the right side for the extra y-axes.
    right_additive = 0
    if extra_ys > 0:
        temp = 0.85
        if extra_ys <= 2:
            temp = 0.75
        elif extra_ys <= 4:
            temp = 0.6
        if extra_ys > 5:
            print('you are being ridiculous')
        fig.subplots_adjust(right=temp)
        right_additive = (0.98 - temp) / float(extra_ys)
    # Move the last y-axis spine over to the right by x% of the width of the axes
    i = 1.
    for ax in axes[2:]:
        ax.spines['right'].set_position(('axes', 1. + right_additive * i))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        ax.yaxis.set_major_formatter(OldScalarFormatter())
        i += 1.
    # To make the border of the right-most axis visible, we need to turn the frame
    # on. This hides the other plots, however, so we need to turn its fill off.

    cols = []
    lines = []
    line_styles = cycle(['-', '-', '-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>',
                         '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])
    colors = cycle([p['color'] for p in list(matplotlib.rcParams['axes.prop_cycle'])])
    for i, (ax, y) in enumerate(zip(axes, ys)):
        ls = next(line_styles)
        if len(y) == 1:
            col = y[0]
            cols.append(col)
            color = next(colors)
            lines.append(ax.plot(df[col], linestyle=ls, label=col, color=color))
            ax.set_ylabel(col, color=color)
            # ax.tick_params(axis='y', colors=color)
            ax.spines['right'].set_color(color)
        else:
            for col in y:
                color = next(colors)
                lines.append(ax.plot(df[col], linestyle=ls, label=col, color=color))
                cols.append(col)
            ax.set_ylabel(' // '.join(y))
            # ax.tick_params(axis='y')
        if invert[i]:
            ax.invert_yaxis()
        if log_scale[i]:
            ax.set_yscale('log')
    axes[0].set_xlabel(df.index.name)
    lns = lines[0]
    for l in lines[1:]:
        lns += l
    labs = [l.get_label() for l in lns]
    axes[0].legend(lns, labs, loc=2)

    if title:
        plt.title(title)
    else:
        plt.title(list(itertools.chain(*ys))[0])
    if save_addr:
        fig.savefig(save_addr)
        print('Saved figure to {0}'.format(save_addr))
    if show:
        print("Showing Plot...")
        plt.show()


def get_obj_name(o) -> str:
    return [k for k, v in locals().items() if v is o][0]


def reverse_enumerate(l):
    for index in reversed(range(len(l))):
        yield index, l[index]


def decimal_to_date(d: str):
    year = int(float(d))
    frac = float(d) - year
    base = dt(year, 1, 1)
    result = base + td(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * frac)
    return result


class DataSource:
    def __init__(self, code: str, provider: Union[str, Callable], rerun: bool=False, start_dt: str='', end_dt: str=''):
        if not start_dt:
            start_dt = '1920-01-01'
        if not end_dt:
            end_dt = dt.today().strftime('%Y-%m-%d')

        self.code = code
        self.data = pd.Series()
        self.rerun = rerun
        self.start_dt = start_dt
        self.end_dt = end_dt

        if provider in ('fred', 'yahoo', 'quandl', 'schiller', 'eod_hist', 'bls', 'worldbank'):
            self.provider = provider
        elif isfunction(provider):
            self.provider = provider
        else:
            raise ValueError('Unrecognized data source provider passed to new data source.')

    def set_data(self, data: pd.DataFrame):
        self.data = data

    def collect_data(self):

        load_env()
        if isfunction(self.provider):
            if self.code:
                self.data = self.provider(self.code)
            else:
                self.data = self.provider()

        elif self.provider == 'fred':
            fred = Fred(api_key=os.environ['TOKEN_FRED'])
            self.data = fred.get_series(self.code, observation_start=self.start_dt, observation_end=self.end_dt)

        elif self.provider == 'eod_hist':
            url = 'https://eodhistoricaldata.com/api/eod/{0}'.format(self.code)
            params = {'api_token': os.environ['TOKEN_EODHIST']}
            expire_after = td(days=1).total_seconds()
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
            url = 'http://www.econ.yale.edu/~shiller/data/ie_data_with_TRCAPE.xls'
            webpage = requests.get(url, stream=True)
            self.data = pd.read_excel(io.BytesIO(webpage.content), 'Data', header=7, skipfooter=1)
            self.data.index = self.data['Date'].apply(lambda x: dt.strptime(str(x).format(x, '4.2f'), '%Y.%m'))
            self.data = self.data[self.code]
            print(self.data.tail(5))

        elif self.provider == 'quandl':
            self.data = quandl.get(self.code,
                                   authtoken=os.environ['TOKEN_QUANDL'],
                                   collapse="quarterly",
                                   start_date=self.start_dt,
                                   end_date=self.end_dt)['Value']

        elif self.provider == 'bls':
            self.data = bls.get_series([self.code],
                                       startyear=dt.strptime(self.start_dt, '%Y-%m-%d').year,
                                       endyear=dt.strptime(self.end_dt, '%Y-%m-%d').year,
                                       key=os.environ['TOKEN_BLS']
                                       )

        elif self.provider == 'worldbank':
            self.data = wbdata.get_data(self.code,
                                        country='US',
                                        data_date=(dt.strptime(self.start_dt, '%Y-%m-%d'),
                                                   dt.strptime(self.end_dt, '%Y-%m-%d')),
                                        convert_date=True,
                                        pandas=True,
                                        keep_levels=False
                                        )
            print(self.data.tail(5))

        print("Collected data for [{0}]".format(self.code))


def shift_x_quarters(s: pd.Series, y: int):
    return s / s.shift(y)


# def impute_if_any_nulls(impute_df: pd.DataFrame, imputer: str = 'knnimpute', verbose: bool=False):
#     impute_names = impute_df.columns.values.tolist()
#     impute_index = impute_df.index.values
#     if impute_df.isnull().any().any():
#         print('Running imputation')
#         try:
#             if imputer == 'knnimpute':
#                 raise ValueError('knnimpute requested')
#             imputer = importlib.import_module("fancyimpute")
#
#             solver = imputer.BiScaler()
#             # solver = imputer.NuclearNormMinimization()
#             # solver = imputer.MatrixFactorization()
#             # solver = imputer.IterativeSVD()
#             impute_df = solver.fit_transform(impute_df.values)
#         except (ImportError, ValueError) as e:
#             imputer = importlib.import_module("knnimpute")
#             impute_df = imputer.knn_impute_few_observed(impute_df.astype(float).values,
#                                                         missing_mask=impute_df.isnull().values,
#                                                         k=5,
#                                                         verbose=verbose)
#         # df = solver.complete(df.values)
#
#     impute_df = pd.DataFrame(data=impute_df, columns=impute_names, index=impute_index)
#     for n in impute_names.copy():
#         if impute_df[n].isnull().any().any():
#             print('Field [{0}] was still empty after imputation! Removing it!'.format(n))
#             impute_names.remove(n)
#
#     return impute_df, impute_names

def get_operator_fn(op):
    return {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '%': operator.mod,
        '^': operator.xor,
    }[op]


def permutations_with_replacement(n, k):
    for p in itertools.product(n, repeat=k):
        yield p


# Construct 2 different level 1 interactions between all x-variables:
#   1. Multiply variable 1 by variable 2
#   2. Divide variable 1 by variable 2
def get_level1_interactions(df: pd.DataFrame, x_names: list, min: float = 0.1, max: float = 0.9):
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


#  _trend_rise_flag: Signals when the raw variable value has risen above a certain stdev amount (which is derived using
#    the stdev_qty value), anchored on an ewma of the raw variable (which is derived using the value for alpha)
# _trend_fall_flag: The same as above, but for whenever the trend falls below the stdev amount
def get_diff_std_and_flags(s: pd.Series
                           , field_name: str
                           , alpha: float = 0.125
                           , stdev_qty: float = 2.) \
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


# Derives variables that measure the "strength" of the trend. It does this by calculating 2 ewmas for each of range_len
#   different alpha values. One of the alphas (h2) will always be a bit smaller than the other, so that it will reflect
#   a longer-term weighted trend than the other (h1). It then divides those by eachother. If in many of the range_len
#   cases, the trend is positive, the trend will be quite strong, and will show a strongly positive value. If in many
#   of the range_len cases (each having a greater/lesser recency bias than the others) the trend is negative, then the
#   variable will show a negative value. The "_weighted" version of this variable, which is also calculated, applies
#   weights to each of the range_len variables rather than taking a simple sum of them, so that it favors the longer-
#   term trends over the shorter-term ones, since it is harder for an EWMA to stay positive over the long term vs. over
#   the short term.
def get_trend_variables(s: pd.Series,
                        field_name: str,
                        alpha: float = 0.25) \
        -> (pd.DataFrame, list):
    d = s.to_frame(name=field_name)

    ema_df = pd.DataFrame()
    new_name_trend_strength = field_name + '_trend_strength'
    new_name_trend_strength_weighted = field_name + '_trend_strength_weighted'

    range_len = 17
    index_range = range(1, range_len)
    for exp in index_range:
        h1 = alpha * (1 / (exp))
        h2 = alpha * (1 / (exp + 1))
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
    d[new_name_trend_strength] = ((pos - neg) + range_len) / (2 * range_len)  # creates an index, 0-1, of trend strength
    return d, [new_name_trend_strength, new_name_trend_strength_weighted]


# This algorithm takes a variable and calculates its EWMA. Then it divides the variable by that EWMA to derive a
#   value representing the departure of that variable from the EWMA.
def get_std_from_ewma(s: pd.Series, field_name: str, alpha: float = 0.125) -> (pd.DataFrame, list):
    d = s.to_frame(name=field_name)

    new_name_ewma = field_name + '_ewma'
    new_name_std = field_name + '_std'
    new_name_std_from_ewma = field_name + '_std_from_ewma'

    d[new_name_ewma] = d[field_name].ewm(alpha=alpha).mean()
    d[new_name_std] = d[field_name].ewm(alpha=alpha).std(bias=True)
    d[new_name_std_from_ewma] = (d[field_name] - d[new_name_ewma]) / d[new_name_std]
    return d, [new_name_std_from_ewma]


# This algorithm will estimate the type of trend the variable exhibits (linear, log, or exponential)
#   it will take the trend it selects, fit a line for that trend to the model, and divide the variable
#   by that trendline to get the distance between the variable and its long-term trendline.
def get_diff_from_trend(s: pd.Series, show_plot: bool = False) -> (pd.DataFrame, list):
    # if s.min() < 0:
    #     s = s - s.min()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(1, 2))
    s = pd.Series(data=[v[0] for v in scaler.fit_transform(s.values.reshape(-1, 1))],
                  name=s.name, index=s.index)

    N = len(s.values)
    from sklearn import linear_model
    n = pd.Series([v + 2 for v in range(len(s.values))], index=s.index)
    logn = np.log(n)

    # fit linear model
    linmod = linear_model.LinearRegression()
    x = np.reshape(n.values, (N, 1))
    y = s
    linmod.fit(x, y)
    linmod_rsquared = linmod.score(x, y)
    m = linmod.coef_[0]
    c = linmod.intercept_
    linear = c + (n * m)

    # fit log-log model
    loglogmod = linear_model.LinearRegression()
    x = np.reshape(logn.values, (N, 1))
    y = np.log(s)
    loglogmod.fit(x, y)
    loglogmod_rsquared = loglogmod.score(x, y)
    m = loglogmod.coef_[0]
    c = loglogmod.intercept_
    polynomial = math.exp(c) * np.power(n, m)

    # fit log model
    logmod = linear_model.LinearRegression()
    x = np.reshape(n.values, (N, 1))
    y = np.log(s)
    logmod.fit(x, y)
    logmod_rsquared = logmod.score(x, y)
    m = logmod.coef_[0]
    c = logmod.intercept_
    exponential = np.exp(n * m) * math.exp(c)

    if show_plot:
        # plot results
        plt.subplot(1, 1, 1)
        plt.plot(n, s, label='series {0}'.format(s.name), lw=2)

        # linear model
        m = linmod.coef_[0]
        c = linmod.intercept_
        plt.plot(n, linear, label='$t={0:.1f} + n*{{{1:.1f}}}$ ($r^2={2:.4f}$)'.format(c, m, linmod_rsquared),
                 ls='dashed', lw=3)

        # log-log model
        m = loglogmod.coef_[0]
        c = loglogmod.intercept_
        plt.plot(n, polynomial,
                 label='$t={0:.1f}n^{{{1:.1f}}}$ ($r^2={2:.4f}$)'.format(math.exp(c), m, loglogmod_rsquared),
                 ls='dashed', lw=3)

        # log model
        m = logmod.coef_[0]
        c = logmod.intercept_
        plt.plot(n, exponential,
                 label='$t={0:.1f}e^{{{1:.1f}n}}$ ($r^2={2:.4f}$)'.format(math.exp(c), m, logmod_rsquared),
                 ls='dashed', lw=3)

        # Show the plot results
        plt.legend(loc='upper center', prop={'size': 16}, borderaxespad=0., bbox_to_anchor=(0.5, 1.25))
        plt.show()

    max_rsq = round(max(linmod_rsquared, logmod_rsquared, loglogmod_rsquared), 2)
    if linmod_rsquared + 0.1 >= max_rsq:
        trend = linear
    elif loglogmod_rsquared + 0.05 >= max_rsq:
        trend = polynomial
    else:
        trend = exponential

    new_name_vs_trend = s.name + '_vs_trend'
    d = pd.DataFrame(s / trend, columns=[new_name_vs_trend])
    return d, [new_name_vs_trend]


def time_since_last_true(s: pd.Series) -> pd.Series:
    s.iloc[0] = prev_val = int(round(s.value_counts()[False] / 2 / s.value_counts()[True], 0))
    for i, v in list(s.iteritems())[1:]:
        if v:
            s.at[i] = 0
        else:
            s.at[i] = prev_val + 1
        prev_val = s.at[i]
    return s.astype(int)


def vwma(vals: pd.Series, mean_alpha: float = 0.125, verbose: bool = False, inverse: bool = False):
    orig_idx = vals.index
    diff_vals = vals / vals.shift(1)
    if verbose:
        print(diff_vals)
        print(len(diff_vals))
    diff_vals.dropna(inplace=True)
    scaler_std = sk_prep.StandardScaler()
    # normal_vol_ewma = vals.ewm(alpha=mean_alpha).std()
    # if verbose:
    #     print(normal_vol_ewma)
    normal_vol_ewma = [v[0] for v in scaler_std.fit_transform(diff_vals.values.reshape(-1, 1))]
    if inverse:
        normal_vol_ewma = [1 - logistic.cdf(v) for v in normal_vol_ewma]
    else:
        normal_vol_ewma = [logistic.cdf(v) for v in normal_vol_ewma]

    avg_ewm_factor = mean_alpha / 0.5
    alphas = [v * avg_ewm_factor for v in normal_vol_ewma]
    alphas = [mean_alpha] + alphas
    if verbose:
        print('Length of alphas list: ', len(alphas))
        print('Length of values list: ', len(vals))
    final_data = pd.DataFrame(data=list(zip(vals, alphas)), columns=['vals', 'alpha'], index=orig_idx)
    cume_alphas = None
    last_vwma = None
    for idx, val, alpha in final_data.itertuples():
        if not cume_alphas:
            cume_alphas = mean_alpha
            vwma = val
        else:
            cume_alphas += (alpha * (1 - cume_alphas))
            adj_alpha = alpha / cume_alphas
            vwma = (val * adj_alpha) + (last_vwma * (1 - adj_alpha))
        final_data.at[idx, 'cume_alphas'] = cume_alphas
        final_data.at[idx, 'vwma'] = vwma
        last_vwma = vwma
        # print(val, alpha)

    # print(sum(normal_vol_ewma)/len(normal_vol_ewma))
    if verbose:
        print('==== Head ====')
        print(final_data.head(10))
        print('==== Tail ====')
        print(final_data.tail(10))
        print(len(final_data['vwma']))

    # final_data.set_index(orig_idx)
    return final_data['vwma']


def get_level1_correlations(df: pd.DataFrame, x_names: list, top_n: int = -1, ewm_alpha: int=0.125):
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
def get_all_interactions(df, x_names: list, curr_name: str = None, series: pd.Series = None):
    new_x_names = []
    if x_names:
        next_name = x_names[0]
        if not curr_name:
            get_all_interactions(new_names=x_names[1:], curr_name=next_name, series=df[next_name])
        else:
            if abs(np.corrcoef(series, df[next_name])[0][1]) <= 0.3:
                for op in ['*', '/']:
                    next_curr_name = '{0}_{1}_{2}'.format(curr_name, op, next_name)
                    next_series = get_operator_fn(op)(series, df[next_name].replace({0: np.nan}))
                    new_x_names.append(next_curr_name)
                    df[next_curr_name] = next_series
                    print("Adding interaction term: {0}".format(next_curr_name))
                    get_all_interactions(new_names=x_names[1:], curr_name=next_curr_name, series=next_series)
            else:
                get_all_interactions(new_names=x_names[1:], curr_name=curr_name, series=series)
        get_all_interactions(new_names=x_names[1:])
    return df, new_x_names

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


def reduce_vars_corr(df: pd.DataFrame, field_names: list, max_num: float, imputer: str='knnimpute'):
    num_vars = len(field_names) - 1
    print('Current vars:  {0}'.format(num_vars))
    if not max_num or max_num < 1:
        if max_num == 0:
            max_num = 0.5
        max_num = int(np.power(df.shape[0], max_num))

    print('Max allowed vars: {0}'.format(max_num))

    if num_vars > max_num:

        if df.isnull().any().any():
            imputed_df, field_names = impute_if_any_nulls(df.loc[:, field_names].astype(float))
            for n in field_names:
                df[n] = imputed_df[n]
        # Creates Correlation Matrix
        corr_matrix = df.loc[:, field_names].corr()

        max_corr = [(fld, corr_matrix.iloc[i + 1, :i].max()) for i, fld in reverse_enumerate(field_names[1:])]
        max_corr.sort(key=lambda tup: tup[1])

        return_x_vals = [fld for fld, corr in max_corr[:max_num]]
        print('Number of Remaining Fields: {0}'.format(len(return_x_vals)))
        print('Remaining Fields: {0}'.format(return_x_vals))
        return df, return_x_vals

    return df, field_names


def reduce_vars_pca(df: pd.DataFrame, field_names: list, max_num: Union[int, float] = None):
    num_vars = len(field_names) - 1
    print('Current vars: {0}'.format(num_vars))
    if not max_num or max_num < 1:
        if max_num == 0:
            max_num = 0.5
        max_num = int(np.power(df.shape[0], max_num))

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
        if v in drop_cols:
            continue

        i2, c = sorted([(i2 + 1, v2) for i2, v2 in enumerate(corr_matrix.iloc[i, i + 1:])], key=lambda tup: tup[1])[-1]

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


def remove_high_vif(X: pd.DataFrame, max_num: Union[int, float] = None):
    num_vars = X.shape[1]
    colnames = X.columns.values
    if not max_num or max_num == 0:
        max_num = round(np.power(X.shape[0], 0.3), 0)

    if num_vars > max_num:
        print('Removing variables with high VIF. New variable qty will be: [{0}]'.format(max_num))

        # from joblib import Parallel, delayed
        while num_vars > max_num:
            # vif = Parallel(n_jobs=-1, verbose=-1)(delayed(variance_inflation_factor)(X.loc[:, colnames].values, ix) for ix in range(X.loc[:, colnames].shape[1]))
            vif = [variance_inflation_factor(X.loc[:, colnames].values, ix) for ix in
                   range(X.loc[:, colnames].shape[1])]

            maxloc = vif.index(max(vif))
            print('dropping \'' + X.loc[:, colnames].columns[maxloc] + '\' at index: ' + str(maxloc))
            del colnames[maxloc]

        print('Remaining variables:')
        print(colnames)

    return X.loc[:, colnames], colnames.tolist()


def calc_equity_alloc(start_dt: str='', end_dt: str='') -> pd.Series:

    if not start_dt:
        start_dt = '1920-01-01'
    if not end_dt:
        end_dt = dt.today().strftime('%Y-%m-%d')

    fred = Fred(api_key=os.environ['TOKEN_FRED'])
    nonfin_biz_equity_liab = fred.get_series('NCBEILQ027S', observation_start=start_dt, observation_end=end_dt)
    nonfin_biz_credit_liab = fred.get_series('BCNSDODNS', observation_start=start_dt, observation_end=end_dt)
    household_nonprofit_credit_liab = fred.get_series('CMDEBT', observation_start=start_dt, observation_end=end_dt)
    fedgov_credit_liab = fred.get_series('FGSDODNS', observation_start=start_dt, observation_end=end_dt)
    localgov_ex_retirement_credit_liab = fred.get_series('SLGSDODNS', observation_start=start_dt,
                                                         observation_end=end_dt)
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
    df = pd.read_table(strio, sep='\\t', skiprows=3, engine='python')

    df['End of month'] = pd.DatetimeIndex(pd.to_datetime(df['End of month']),
                                          dtype=dt.date).to_period('M').to_timestamp('M')
    df.set_index(['End of month'],
                 drop=True,
                 inplace=True,
                 verify_integrity=True)

    df = df \
        .replace('[\$,)]', '', regex=True) \
        .replace('[(]', '-', regex=True) \
        .astype(float)

    # print(df)
    return make_qtrly(df[field_name], 'first')


def pickle_load(name: str, dir: str = None) -> object:
    if '.p' not in name:
        name += '.p'

    if dir:
        name = os.path.join(dir, name)

    with open(name, 'rb') as f:
        data = pickle.load(f)

    return data


def pickle_dump(data: object, name: str, dir: str = None):
    if '.p' not in name:
        name += '.p'

    if dir:
        name = os.path.join(dir, name)

    with open(name, 'wb') as f:
        pickle.dump(data, f)
