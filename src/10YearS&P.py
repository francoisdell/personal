import calendar
import math
import pickle
from datetime import datetime

import fancyimpute
import matplotlib
import numpy as np
import pandas as pd
import quandl
import bls
# from matplotlib import rcParams
import yahoo_finance
from fredapi import Fred
from matplotlib import pyplot as plt
from matplotlib.ticker import OldScalarFormatter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from inspect import isfunction

import Model_Builder as mb

load_from_file = True
from collections import OrderedDict

selection_limit = 1.0e-2
train_pct = 0.8
start_dt = '1952-01-01'
end_dt = datetime.today().strftime('%Y-%m-%d')

real = False
sp_field_name = 'sp500'
if real:
    sp_field_name += '_real'

recession_field_name = 'recession_usa'

#use the list to filter the local namespace
safe_func_list = ['first', 'last', 'mean']
safe_funcs = dict([(k, locals().get(k, None)) for k in safe_func_list])

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

def get_closes(d: list, fld_names: list=['Close','col3']) -> pd.Series:
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
def chart(d, ys, invert, log_scale):

    from itertools import cycle
    fig, ax = plt.subplots()

    axes = [ax]
    for y in ys[1:]:
        # Twin the x-axis twice to make independent y-axes.
        axes.append(ax.twinx())

    extra_ys = len(axes[2:])

    # Make some space on the right side for the extra y-axes.
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

    plt.show()

def get_obj_name(o) -> str:
    return [k for k, v in locals().items() if v is o][0]

def reverse_enumerate(l):
   for index in reversed(range(len(l))):
      yield index, l[index]

def predict_returns(df: pd.DataFrame, x_names: list, y_field_name: str, years_forward: int, prune: bool=False)\
        -> OrderedDict:

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
    df = mb.predict(df
                    , x_fields=x_fields
                    , y_field=y_field
                    , model_type='ridge'
                    , report_name='sp500_ridge'
                    , show_model_tests=True
                    , retrain_model=True
                    , selection_limit=selection_limit
                    , predict_all=True
                    , verbose=False
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
          , log_scale=[False, True])

    return OrderedDict(((forward_y_field_name, df[forward_y_field_name])
        , (forward_y_field_name_pred, df[forward_y_field_name_pred])
        , (y_field_name_pred, df[y_field_name_pred])))



def predict_recession(df: pd.DataFrame, x_names: list, y_field_name: str, years_forward: int, prune: bool=False) \
        -> OrderedDict:

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
    df = mb.predict(df
                    , x_fields=x_fields
                    , y_field=y_field
                    , model_type='gbc'
                    , report_name='recession_gbc'
                    , show_model_tests=True
                    , retrain_model=True
                    , selection_limit=selection_limit
                    , predict_all=True
                    , verbose=False
                    , train_pct=train_pct
                    , random_train_test=False)

    forward_y_field_name_pred = 'pred_' + forward_y_field_name
    #################################

    print(max(df.index))
    from dateutil.relativedelta import relativedelta
    df = df.reindex(
        pd.DatetimeIndex(start=df.index.min(), end=max(df.index) + relativedelta(years=years_forward), freq='1Q'))

    y_field_name_pred = '{0}_pred'.format(y_field_name)
    df[y_field_name_pred] = df[forward_y_field_name_pred].shift(years_forward * 4)

    # if years_forward in [10]:
    chart(df
          , ys=[['sp500'], [y_field_name, y_field_name_pred]]
          , invert=[False, False]
          , log_scale=[True, False])

    return OrderedDict(((forward_y_field_name, df[forward_y_field_name])
        , (forward_y_field_name_pred, df[forward_y_field_name_pred])
        , (y_field_name_pred, df[y_field_name_pred])))

def predict_ensemble(df: pd.DataFrame, x_names: list, y_field_name: str, prune: bool=False):

    train_mask = ~df[y_field_name].isnull()
    val_mask = df[y_field_name].isnull() & ~df[x_names].isnull().any(axis=1)

    model_list = ['ridge']

    if df[x_names].isnull().any().any():
        print('Running imputation')
        solver = fancyimpute.MICE(init_fill_method='random') # mean, median, or random
        # solver = fancyimpute.NuclearNormMinimization()
        # solver = fancyimpute.MatrixFactorization()
        # solver = fancyimpute.IterativeSVD()
        df.loc[:, x_names] = solver.complete(df.loc[:, x_names].values)

    result_field_dict = OrderedDict()
    #################################
    # USING THE MODEL BUILDER CLASS #
    #################################
    y_field = OrderedDict([(y_field_name, 'num')])
    x_fields = OrderedDict([(v, 'num') for v in x_names])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.to_string())

    for model_name in model_list:
        report_name = '{0}_{1}'.format(y_field_name, model_name)
        df = mb.predict(df
                        , x_fields=x_fields
                        , y_field=y_field
                        , model_type=model_name
                        , report_name=report_name
                        , show_model_tests=True
                        , retrain_model=True
                        , selection_limit=selection_limit
                        , predict_all=True
                        , verbose=False
                        , train_pct=train_pct
                        , random_train_test=False
                        )

        y_field_name_pred = 'pred_{0}'.format(report_name)
        df.rename(columns={'pred_' + y_field_name: y_field_name_pred}, inplace=True)

        result_field_dict[y_field_name_pred] = df[y_field_name_pred]

    #################################

    df = df.reindex(pd.DatetimeIndex(start=df.index.min(), end=max(df.index), freq='1Q'))
    chart(df
          , ys=[[y_field_name, *list(result_field_dict.keys())]]
          , invert=[False]
          , log_scale=[True])

    return result_field_dict

fred = Fred(api_key='b604ef6dcf19c48acc16461e91070c43')

class data_source:
    def __init__(self, code: str, provider: str):
        self.code = code
        self.data = None

        if provider in ('fred', 'yahoo', 'quandl'):
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
            self.data = fred.get_series(self.code, observation_start=start_dt, observation_end=end_dt)
        elif self.provider == 'yahoo':
            data_obj = yahoo_finance.Share(self.code)
            self.data = data_obj.get_historical(start_date=start_dt, end_date=end_dt)
        elif self.provider == 'quandl':
            self.data = quandl.get(self.code, authtoken="xg_fvD6FLD_qzg2Mc5z-"
                       , collapse="quarterly", start_date=start_dt, end_date=end_dt)['Value']
        elif self.provider == 'bls':
            self.data = bls.get_series([self.code]
                                , startyear=datetime.strptime(start_dt, '%Y-%m-%d').year
                                , endyear=datetime.strptime(end_dt, '%Y-%m-%d').year
                                , key='3d75f024d5f64d189e5de4b7cbb99730'
                                )

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

data_sources = dict()
data_sources['cpi_urb_nonvol'] = data_source('CPILFESL', 'fred')
data_sources['netexp_nom'] = data_source('NETEXP', 'fred')
data_sources['gdp_nom'] = data_source('GDP', 'fred')
data_sources['sp500'] = data_source('^GSPC', 'yahoo')
data_sources['tsy_3mo_yield'] = data_source('^IRX', 'yahoo')
data_sources['tsy_5yr_yield'] = data_source('^FVX', 'yahoo')
data_sources['tsy_10yr_yield'] = data_source('^TNX', 'yahoo')
data_sources['tsy_30yr_yield'] = data_source('^TYX', 'yahoo')
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
data_sources['recession_usa'] = data_source('USREC', 'fred')
ds_names = [k for k in data_sources.keys()]

try:
    if load_from_file:
        with open('src/sp500_hist.p', 'rb') as f:
            (data_sources_temp,) = pickle.load(f)
            for k in data_sources.keys():
                if k not in data_sources_temp:
                    print('New data source added: [{0}]'.format(k))
                    data_sources_temp[k] = data_sources[k]
        data_sources = data_sources_temp
        for k, ds in data_sources.items():
            if ds.data is None:
                ds.collect_data()
            data_sources[k] = ds

    else:
        raise ValueError('Per Settings, Reloading Data From Yahoo Finance/FRED/Everything Else.')
except Exception as e:
    print(e)
    for k, ds in data_sources.items():
        ds.collect_data()
        data_sources[k] = ds

with open('src/sp500_hist.p', 'wb') as f:
    pickle.dump((data_sources,), f)


from statsmodels import robust
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

    # modified_z_score = 0.6745 * abs_dev / y_mad
    # modified_z_score[y == m] = 0
    # return modified_z_score > thresh


# from pandas_datareader import data, wb
# lol = data.DataReader(['TSLA'], 'yahoo', start_dt, end_dt)

df = pd.DataFrame()
for k, ds in data_sources.items():
    if ds.provider == 'yahoo':
        df[k] = make_qtrly(get_closes(ds.data), 'last')
    elif ds.provider == 'fred':
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

# 4.2655 corresponds to a 0.125 weight
def convert_to_ewma(df: pd.DataFrame, field_name: str, groupby_fields: list=lambda x: True, ewm_halflife: float=4.2655/3):
    ewma_field_name = field_name + '_ewma'
    df[ewma_field_name] = df[field_name].groupby(by=groupby_fields).apply(lambda x: x.ewm(halflife=ewm_halflife).mean())
    return df, ewma_field_name

x_names = [
    'equity_alloc'
    , 'tsy_10yr_yield'
    , 'tsy_5yr_yield'
    , 'tsy_3mo_yield'
    # , 'diff_tsy_10yr_and_cpi' # Makes the models go FUCKING CRAZY
    , 'unempl_rate'
    # , 'empl_construction'
    , 'sp500_peratio'
    , 'capacity_util_mfg'
    , 'capacity_util_chem'
    # , 'gold_fix_3pm'
    # , 'fed_funds_rate'
    # , 'tsy_3m10y_curve'
    # , 'industrial_prod'
    # , 'tsy_10yr_minus_fed_funds_rate'
    # , 'tsy_10yr_minus_cpi'
    # , 'netexp_pct_of_gdp' # Will cause infinite values when used with SHIFT (really any y/y compare)
    # , 'gdp_nom'
    # , 'netexp_nominal' # Will cause infinite values when used with SHIFT (really any y/y compare)
    # , 'base_minus_fed_res_adj' # May also make the models go FUCKING CRAZY # Not much history
    # , 'tsy_30yr_yield' # Not much history
    , 'med_family_income_vs_house_price'
    # , 'pers_savings_rt'
    ]


empty_cols = [c for c in df.columns.values if all(df[c].isnull())]
if len(empty_cols) > 0:
    raise Exception("Empty columns in final dataframe: {0}".format(empty_cols))

for v in x_names:
    min_dt = df[v][~df[v].isnull()].index.values[0]
    print('Earliest date for series [{0}] is: {1}'.format(v, min_dt))


# If you d
# non_null_mask = ~pd.isnull(df.loc[:, x_names]).all(axis=1)
non_null_mask = [x >= 3 for x in df.loc[:, x_names].count(axis=1).values]
# ensemble_mask = ~pd.isnull(df.loc[:, x_names]).any(axis=1)
df = df.loc[non_null_mask, :]

def shift_x_years(s: pd.Series, y: int):
    return s / s.shift(4*y)

def impute_if_any_nulls(df):
    if df.isnull().any().any():
        print('Running imputation')
        solver = fancyimpute.MICE(init_fill_method='random')  # mean, median, or random
        # solver = fancyimpute.NuclearNormMinimization(error_tolerance=0.001)
        # solver = fancyimpute.MatrixFactorization()
        # solver = fancyimpute.IterativeSVD()
        df = solver.complete(df.values)
    return df

print('Adding x-year diff terms.')
diff_x_names = [
    'gdp_nom'
    , 'cpi_urb_nonvol'
    , 'empl_construction'
    , 'industrial_prod'
    , 'housing_starts'
    , 'housing_supply'
    , 'med_house_price'
    , 'med_family_income'
    , 'unempl_rate'
    , 'real_med_family_income'
    , 'combanks_business_loans'
    , 'combanks_assets_tot'
    , 'mortage_debt_individuals'
    , 'real_estate_loans'
    , 'foreign_dir_invest'
    # , 'gross_savings'
    , 'tax_receipts_corp'
    # , 'fed_funds_rate'
    # , 'gold_fix_3pm'
]

# Interactions between a field and its previous values
for name in diff_x_names:
    for y in [1]:
        diff_field_name = '{0}_{1}yr_diff'.format(name, y)
        df[diff_field_name] = shift_x_years(df[name], y)
        x_names.append(diff_field_name)

# IMPUTE VALUES!!!
df.loc[:, x_names] = impute_if_any_nulls(df.loc[:, x_names])

def convert_to_pca(pca_df: pd.DataFrame, field_names: list):
    from sklearn.decomposition import TruncatedSVD
    pca_model = TruncatedSVD(n_components=4, random_state=555)

    x_results = pca_model.fit_transform(pca_df.loc[:, field_names]).T
    print(pca_model.components_)
    print(pca_model.explained_variance_ratio_)

    x_names_pca = []
    for idx, component in enumerate(x_results):
        pca_name = 'pca_{0}'.format(idx)
        pca_df[pca_name] = component
        x_names_pca.append(pca_name)

    return x_names_pca

# UNCOMMENT IF YOU WANT TO UTILIZE PCA
x_names = convert_to_pca(df, x_names)

import operator
def get_operator_fn(op):
    return {
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '/' : operator.truediv,
        '%' : operator.mod,
        '^' : operator.xor,
        }[op]

import itertools

def permutations_with_replacement(n, k):
    for p in itertools.product(n, repeat=k):
        yield p

import random
print('Adding interaction terms.')
new_x_names = []
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

get_all_interactions(x_names)
x_names.extend(new_x_names)

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

# print('Adding interaction terms.')
# new_x_names = []
# for k, v in enumerate(x_names[:-1]):
#     for v1 in x_names[k+1:]:
#         corr = np.corrcoef(df[v], df[v1])[0][1]
#         if abs(corr) <= 0.75:
#             # Interactions between two different fields generated through multiplication
#             interaction_field_name = '{0}_times_{1}'.format(v, v1)
#             df[interaction_field_name] = df[v] * df[v1]
#             new_x_names.append(interaction_field_name)
#
#             # Interactions between two different fields, generated through division
#             interaction_field_name = '{0}_div_{1}'.format(v, v1)
#             df[interaction_field_name] = df[v] / df[v1].replace({0: np.nan})
#             new_x_names.append(interaction_field_name)
# x_names.extend(new_x_names)

print('Converting fields to EWMA fields.')
new_x_names = []
for v in x_names:
    df, ewma_field_name = convert_to_ewma(df, v)
    new_x_names.append(ewma_field_name)
x_names = new_x_names

print('Converting fields to trimmed fields.')
new_x_names = []
for v in x_names:
    new_x_name = v + '_trim'
    df[new_x_name] = trim_outliers(df[v], thresh=4)
    new_x_names.append(new_x_name)
x_names = new_x_names

# IMPUTE VALUES!!!
df.loc[:, x_names] = impute_if_any_nulls(df.loc[:, x_names])

new_x_names = []
operations = [('pow2', math.pow, (2,)), ('sqrt', math.sqrt, None)]
for suffix, op, var in operations:
    for v in x_names:
        new_x_name = '{0}_{1}'.format(v, suffix)
        df[new_x_name] = df[v].abs().apply(op, args=var) * df[v].apply(lambda x: -1 if x < 0 else 1)
        new_x_names.append(new_x_name)
x_names.extend(new_x_names)

def get_diff_std_and_flags(df: pd.DataFrame
                           , field_name: str
                           , groupby_fields: list=lambda x: True
                           , ewm_halflife: float=4.2655
                           , stdev_qty: int=2)\
        -> (pd.DataFrame, list):

    ewma_field_name = field_name + '_ewma'
    if ewma_field_name not in df.columns:
        df, ewma_field_name = convert_to_ewma(df,field_name, groupby_fields)

    new_name_prefix = field_name + '_val_to_ewma'
    new_name_diff = new_name_prefix + '_diff'
    new_name_diff_std = new_name_prefix + '_diff_std'
    new_name_add_std = new_name_prefix + '_add_std'
    new_name_sub_std = new_name_prefix + '_sub_std'
    new_name_trend_fall = new_name_prefix + '_trend_fall_flag'
    new_name_trend_rise = new_name_prefix + '_trend_rise_flag'

    df[new_name_diff] = df[field_name] - df[ewma_field_name]
    df[new_name_diff_std] = df.groupby(by=[groupby_fields])[new_name_diff].apply(lambda x: x.ewm(halflife=ewm_halflife).std(bias=False))
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

df, new_x_names = get_diff_std_and_flags(df, 'sp500')
# x_names.extend(new_x_names)

sp500_qtr_since_last_corr = 'sp500_qtr_since_last_corr'
df[sp500_qtr_since_last_corr] = time_since_last_true(df[new_x_names[-1]])
# x_names.append(sp500_qtr_since_last_corr)
# print('===== Head =====\n', df.head(5))
# print('===== Tail =====\n', df.tail(5))

# print(df.head(5))
# print(df.tail(5))

# df_valid = df.loc[~train_mask, :]
# predict_years_forward = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
predict_years_forward = [9]

ensemble_df = pd.DataFrame(df[sp_field_name], index=df.index, columns=[sp_field_name])
ensemble_x_names = list()
for yf in predict_years_forward:
    d = predict_returns(df, x_names, sp_field_name, yf, prune=True)
    for k,v in d.items():
        df[k] = v

    new_dataset = [v for v in d.values()][-1]
    new_field_name = 'sp500_{0}yr'.format(yf)

    # print(ensemble_df.index)
    # print(new_dataset.index)
    new_df = new_dataset.to_frame(name=new_field_name)
    ensemble_df = ensemble_df.join(new_df, how='right')
    ensemble_x_names.append(new_field_name)

recession_predict_years_forward = [1, 2, 3]
for yf in recession_predict_years_forward:
    d = predict_recession(df, x_names, recession_field_name, yf, prune=True)
    for k,v in d.items():
        df[k] = v

    new_dataset = [v for v in d.values()][-1]
    new_field_name = 'recession_{0}yr'.format(yf)

    # print(ensemble_df.index)
    # print(new_dataset.index)
    new_df = new_dataset.to_frame(name=new_field_name)

# Use the results of the various forward prediction models to construct an ensemble model!!!
ensemble_mask = [x >= 5 for x in ensemble_df.loc[:, ensemble_x_names].count(axis=1).values]
# ensemble_mask = ensemble_mask.values.tolist() + [True for v in range(ensemble_df.shape[0] - non_null_mask.shape[0])]
ensemble_df = ensemble_df.loc[ensemble_mask, :]
# d = predict_ensemble(ensemble_df, ensemble_x_names, sp_field_name)
