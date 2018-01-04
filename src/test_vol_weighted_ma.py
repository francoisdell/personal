from typing import Union
import numpy as np
import pandas as pd
import math
from scipy.stats import logistic
from sklearn import preprocessing as sk_prep
import sklearn


def vwma(vals: pd.Series, mean_alpha: float = 0.125, verbose: bool = False):
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
    normal_vol_ewma = [logistic.cdf(v) for v in normal_vol_ewma]
    avg_ewm_factor = mean_alpha / 0.5
    alphas = [v * avg_ewm_factor for v in normal_vol_ewma]
    alphas = [mean_alpha] + alphas
    if verbose:
        print('Length of alphas list: ', len(alphas))
        print('Length of values list: ', len(vals))
    final_data = pd.DataFrame(data=list(zip(vals, alphas)), columns=['vals', 'alpha'])
    cume_alphas = None
    last_vwma = None
    for idx, val, alpha in final_data.itertuples():
        if idx == 0:
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

    final_data.set_index(orig_idx)
    return final_data['vwma']


normdist_vals = np.random.normal(loc=0.06, scale=0.06, size=50)
curr_val = 1
random_walk_vals = list()
for v in normdist_vals:
    curr_val += curr_val*v
    random_walk_vals.append(curr_val)

# print(random_walk_vals)

vwma(pd.Series(random_walk_vals), verbose=True)

