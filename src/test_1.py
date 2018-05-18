import requests
import pandas as pd
import io

# url = 'http://www.nyxdata.com/nysedata/asp/factbook/table_export_csv.asp?mode=tables&key=50'
# with requests.Session() as s:
#     download = s.get(url=url)
#
# strio = io.StringIO(download.text)
# df = pd.read_table(strio, sep='\\t', skiprows=3)
#
# print(df)
#
# df['End of month'] = pd.DatetimeIndex(pd.to_datetime(df['End of month'])).to_period('M').to_timestamp('M')
# df.set_index(['End of month'], inplace=True, drop=True)
# print(df)
import platform
print(platform.system().lower())

from scipy import sparse
import numpy as np
from typing import Union

def matrix_vstack(m: tuple, return_sparse: bool = None, use_sparse: bool=False):
    if sum([sparse.issparse(d) for d in list(m)]) == 1:
        if use_sparse:
            m = [sparse.csc_matrix(d) if not sparse.issparse(d) else d for d in list(m)]
        else:
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


def matrix_hstack(m: tuple, return_sparse: bool = None, use_sparse: bool=False):
    if sum([sparse.issparse(d) for d in list(m)]) == 1:
        if use_sparse:
            m = [sparse.csc_matrix(d) if not sparse.issparse(d) else d for d in list(m)]
        else:
            m = [d.toarray() if sparse.issparse(d) else (d.reshape(-1, 1) if len(d.shape) == 1 else d) for d in
                 list(m)]
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


def matrix_info(m: sparse.spmatrix):
    print("Shape:", m.shape)
    print(m)


def stack_on_matching_axis(m1: Union[sparse.spmatrix, np.ndarray], m2: sparse.spmatrix, default: str='cols'):
    if m1.ndim == 1:
        m1 = m1.reshape(-1, 1)
    if m2.ndim == 1:
        m2 = m2.reshape(-1, 1)

    if m1.shape == m2.shape:
        choice = default
    elif m1.shape[0] == m2.shape[0]:
        choice = 'cols'
    elif m1.shape[1] == m2.shape[1]:
        choice = 'rows'
    elif m1.shape[0] == m2.shape[1]:
        m2 = m2.transpose()
        choice = 'cols'
    elif m1.shape[1] == m2.shape[0]:
        m2 = m2.transpose()
        choice = 'rows'
    else:
        raise AttributeError('Unable to find an axis on which the two matrices might be stacked')

    if choice == 'cols':
        return matrix_hstack((m1, m2))
    elif choice == 'rows':
        return matrix_vstack((m1, m2))


lol = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
matrix_info(lol)

hah = np.array([11, 12])
matrix_info(hah)

derp = np.vstack((lol, hah))
matrix_info(derp)

lol_sparse = sparse.csc_matrix(lol)

derp_2 = sparse.vstack((lol_sparse, sparse.csc_matrix(hah)))
matrix_info(derp_2)

weee = np.array([11, 12, 13, 14, 15])
matrix_info(weee)

weee_2 = weee.reshape(-1,1)
matrix_info(weee_2)

derp3 = np.hstack((lol, weee_2))
matrix_info(derp3)

derp_3 = sparse.hstack((lol_sparse, sparse.csc_matrix(weee_2)))
matrix_info(derp_3)

print(tuple([1,2]))

stacked_1 = stack_on_matching_axis(lol, hah)
matrix_info(stacked_1)