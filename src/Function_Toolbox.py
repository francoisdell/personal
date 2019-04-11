__author__ = 'Andy Mackenzie'

import logging
import Settings
from chardet.universaldetector import UniversalDetector
from contextlib import contextmanager
import os
import csv
import getopt
import sys
import inspect
from sqlitedict import SqliteDict
import shutil
import signal
import requests
import time
from functools import reduce
import datetime as dt
from datetime import date, timedelta
import pandas as pd
import re
import calendar as cal
import platform
import subprocess
import importlib
import pickle
from re import search as re_search
import math
import json
from typing import Union
import dotenv

preferred_csv_dialect = csv.excel
preferred_csv_dialect.quoting = csv.QUOTE_MINIMAL
preferred_csv_dialect.doublequote = True
preferred_csv_dialect.quotechar = '"'
preferred_csv_dialect.lineterminator = '\n'


def load_env():
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    dotenv.load_dotenv(dotenv_path)


def create_logger(report_name: str):
    s = Settings.Settings(report_name=report_name)
    logfile_addr = s.get_logfile_addr()
    with open(logfile_addr, 'w'):  # Clears the logfile
        pass

    global log
    # create a logger
    try:
        log = logging.getLogger(get_report_name())
    except Exception as e:
        log = logging.getLogger(get_exec_file(whole_path=False))

    log.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(logfile_addr)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    log.addHandler(handler)
    log.info('============ LOG CREATED ============')


def get_logger():
    global log
    try:
        if log:
            return log
        else:
            raise Exception('Log variable does not exist yet.')
    except:
        try:
            create_logger(get_report_name())
        except:
            create_logger(get_exec_file(whole_path=False))
        return log


def print_and_log(msg, log_type: str = None, end='\n'):
    print(msg, end=end)
    l = get_logger()
    if log_type == 'info':
        l.info(msg)
    elif log_type == 'exception':
        l.exception(msg)
    elif log_type == 'error':
        l.error(msg)


def get_exec_file(whole_path: bool = False):
    # Obtains the name of the python file which was originally executed
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__).replace('\\', '/')
    except:
        sFile = sys.executable

    if whole_path:
        sFile = sFile
    else:
        sFile = os.path.splitext(sFile.split('/')[-1])[0]

    return sFile


def get_report_name() -> str:
    return importlib.import_module(get_exec_file()).__file__.split('/')[-1].split('.')[0]


def get_author() -> str:
    return importlib.import_module(get_exec_file()).__author__


def purge_dir(path: str, days: int):
    print_and_log("Scanning %s for files older than %s days." % (path, days), 'info')
    for f in os.listdir(path):

        now = time.time()
        filepath = os.path.join(path, f)
        last_modified = os.stat(filepath).st_mtime
        if last_modified < now - (days * 86400):  # 86400 = number of seconds in one day
            print_and_log("Removing backup older than %s days: %s" % (f, days), 'info')
            try:
                shutil.rmtree(filepath)
                msg = 'DELETED'
            except:
                msg = 'FAILED TO DELETE'
            print_and_log('{0}: {1} (Last modified on: {2})'.format(msg, f, date.fromtimestamp(last_modified)), 'info')


def get_csv_dialect(csv_file: str, encoding: str = '') -> csv.Dialect:
    if not encoding:
        encoding = get_encoding_type(csv_file)
    file = open(csv_file, mode='r', encoding=encoding)
    sniff_sample = file.readline()
    sniffer = csv.Sniffer()
    if sniff_sample.strip():
        try:
            dialect = sniffer.sniff(sniff_sample)
        except csv.Error as e:
            if 'delimiter' in str(e):
                dialect.delimiter = sniffer.preferred[0]
            else:
                raise e
        else:
            if dialect.delimiter not in sniffer.preferred:
                dialect.delimiter = sniffer.preferred[0]
    else:
        print_and_log('No records found in file [{0}]! Assuming dialect of csv.excel.'.format(csv_file))
        dialect = csv.excel

    return dialect


def normalize_csv(csv_file, new_csv_file, encoding: str = None):
    with open(csv_file, mode='r', encoding=encoding) as file:
        dialect = csv.Sniffer().sniff(file.readline())
        dialect.doublequote = True
        data_iter = csv.reader(file, dialect=dialect)
        file.seek(0)
        dialect.quoting = csv.QUOTE_MINIMAL
        dialect.delimiter = ','
        dialect.quotechar = '"'
        dialect.lineterminator = '\n'
        with open(new_csv_file, mode='w', encoding=encoding) as file2:
            data_writer = csv.writer(file2, dialect=dialect)
            data_writer.writerows(data_iter)
    return dialect

def control_dir(dir: str) -> str:
    if "Windows" in platform.architecture()[1]:
        own_cmd = 'takeown /F "%s" /R /d Y' % dir.replace('/', '\\')
        control_cmd = 'icacls "%s" /grant Administrator:(OI)(CI)F /T /Q' % dir.replace('/', '\\')
    else:  # Add something here for Unix/Linux?!?!?
        own_cmd = None
        control_cmd = None

    if own_cmd:
        print("Ownership Command: %s" % own_cmd)
        resp = subprocess.call(own_cmd)

    if control_cmd:
        print("Control Command: %s" % own_cmd)
        resp = subprocess.call(control_cmd)

    return resp


def wnet_connect(fullpath, username, password):

    plat = platform.platform()
    if 'windows' in plat:
        win32wnet = importlib.import_module('win32wnet')
    else:
        raise SystemError("Can't use the win32wnet module on a non-windows platform. ({0})".format(plat))

    netresource = win32wnet.NETRESOURCE()
    netresource.lpRemoteName = fullpath
    try:
        win32wnet.WNetAddConnection2(NetResource=netresource, UserName=username, Password=password, Flags=0)
        print_and_log('Connected to {0}'.format(fullpath))
    except (Exception) as err:
        if isinstance(err, win32wnet.error):
            # Disconnect previous connections if detected, and reconnect.
            if err.winerror == 1219:
                try:
                    win32wnet.WNetCancelConnection2(fullpath, 0, True)
                    win32wnet.WNetAddConnection2(NetResource=netresource, UserName=username, Password=password, Flags=0)
                    pass
                except:
                    print_and_log('Unable to connect! Failing. {0}'.format(fullpath))
                    raise
        else:
            print_and_log('Unable to connect! Failing. {0}'.format(fullpath))
            raise err


def natural_log(x: float) -> float:
    if x > 0:
        return math.exp(x)
    else:
        return math.exp(-x)


def inv_natural_log(x: float) -> float:
    return math.log(x, math.e)


def sigmoid(gamma: float) -> float:
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))


def deep_get(dictionary: Union[dict, SqliteDict], keys: str, default=None) -> dict:
    return reduce(lambda d, key: d.get(key, default) if hasattr(d, 'get') else default,
                  keys.split("/"),
                  dictionary)


def convert_to_ts(val: int) -> dt.datetime:
    if isinstance(val, dt.datetime):
        return val
    try:
        val = str(int(val))
        if len(val) > 10:
            val = val[:10]
        val = dt.datetime.fromtimestamp(int(val))
    except Exception as e:
        try:
            val = pd.to_datetime(val)
        except Exception as e:
            return None
    return val


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def str_as_header(txt: str, char: str='=', end_chars: int=5) -> str:
    end_str = char * (end_chars + 1)
    txt = '{0} {1} {0}'.format(end_str, txt)
    bookends = char * len(txt)
    txt = '\n{0}\n{1}\n{0}'.format(bookends, txt)
    return txt


def regex_get_item(regex: str, value: str, item: int, ignore_errors: bool = True, error_val: object = None):
    result = re_search(regex, value)
    try:
        if result:
            if item == -1:
                item = len(result.groups())
            return result.group(item)
        else:
            raise IndexError('No matches found for the regex string')
    except IndexError as e:
        if ignore_errors:
            return error_val
        else:
            raise e


def reverse_enumerate(l):
    for index in reversed(range(len(l))):
        yield index, l[index]


def prepend_prefix(prefix: str, name: str) -> str:
    if isinstance(prefix, str) \
            and not prefix == name[:len(prefix)]:
        name = prefix + name
    return name


def month_diff(start_dt: dt.datetime, end_dt: dt.datetime) -> float:
    d1_daysinmonth = cal.monthrange(end_dt.year, end_dt.month)[1]
    d2_daysinmonth = cal.monthrange(start_dt.year, start_dt.month)[1]
    diff = (12 * end_dt.year + end_dt.month + (end_dt.day / d1_daysinmonth)) - \
           (12 * start_dt.year + start_dt.month + (start_dt.day / d2_daysinmonth))
    return diff


def get_encoding_type(current_file: str):
    print("Determining file encoding: [%s]" % current_file, 'info')
    detector = UniversalDetector()
    detector.reset()
    for line in open(current_file, 'rb'):
        detector.feed(line)
        if detector.done:
            break
    detector.close()
    print(current_file.split('\\')[-1] + ": " + detector.result['encoding'])
    return detector.result['encoding']


def timer_hms(td: timedelta) -> str:
    m_elapsed, s_elapsed = divmod(td.seconds, 60)
    h_elapsed, m_elapsed = divmod(m_elapsed, 60)
    return "%d:%02d:%02d" % (h_elapsed, m_elapsed, s_elapsed)


def print_status_bar(numerator, denominator, len: int = 25, include_pct: bool = True, include_num: bool = True,
                     start_time: dt.datetime = None):

    # assert numerator <= denominator
    pct = numerator/denominator
    hashtags = math.floor(pct*len)
    dashes = len - hashtags
    bar = 'Progress: '
    if include_pct:
        bar += '{0:.1f}% '.format(pct*100)
    if include_num:
        bar += f'({numerator}/{denominator}) '
    bar += '[{0}]'.format('#' * hashtags + '-' * dashes)
    if start_time:
        if pct > 0:
            elapsed_time = dt.datetime.now() - start_time
            m_elapsed, s_elapsed = divmod(elapsed_time.seconds, 60)
            h_elapsed, m_elapsed = divmod(m_elapsed, 60)
            remaining_time = timedelta(seconds=elapsed_time.seconds * ((1-pct)/pct))
            m_remaining, s_remaining = divmod(remaining_time.seconds, 60)
            h_remaining, m_remaining = divmod(m_remaining, 60)
            bar += ' [Elapsed: {0}] [Remaining: {1}]'\
                .format("%d:%02d:%02d" % (h_elapsed, m_elapsed, s_elapsed),
                        "%d:%02d:%02d" % (h_remaining, m_remaining, s_remaining))

    print(f'\r{bar}', end='' if numerator < denominator else '\n')


def df_agg_from_date(df: pd.DataFrame, group_field, agg_fields, ts_field, joindate_field: str = None,
                     lastxdays: Union[bool, list] = False, firstxdays: Union[bool, list] = False, aggregate: str = 'mean',
                     incl_agg_in_name: Union[bool, str] = True, retain_fields: list = None, presignup: bool = True,
                     postsignup: bool = True, alltime = True, ts_cutoff_field: str = '', curr_dt: dt.datetime = None,
                     include_preceding_period: bool = False, include_following_period: bool = False):

    if not curr_dt:
        curr_dt = df[ts_field].max()  # dt.now()

    new_df = pd.DataFrame(index=df[group_field].unique())
    new_df.index.name = group_field
    suffix = str()
    if isinstance(incl_agg_in_name, str):
        suffix += f'_{incl_agg_in_name}'
    elif incl_agg_in_name:
         suffix += f'_{aggregate}'

    for fld in agg_fields:
        if alltime:
            new_df[f'alltime_{fld}{suffix}'] = df.groupby(by=group_field)[fld].aggregate(aggregate).fillna(0)

        if lastxdays:
            time_periods = dict()
            if lastxdays is True:
                lastxdays = [30, 60, 90]
            for days in lastxdays:
                cutoff_dt = curr_dt - timedelta(days=days)
                time_periods[f'last{days}day_{fld}{suffix}'] = (cutoff_dt, curr_dt)
                if include_preceding_period:
                    preceding_cutoff_dt = cutoff_dt - timedelta(days=days)
                    time_periods[f'last{days}day_preceding{days}_{fld}{suffix}'] = (preceding_cutoff_dt, cutoff_dt)

            for field_name, (start_dt, end_dt) in time_periods.items():
                if ts_cutoff_field:
                    cutoff_series = df[ts_cutoff_field].apply(lambda r: coalesce([r, end_dt]))
                    # take any row with values such that the start date is before the latest date (curr_dr)
                    # and the end date is after the beginning date (start_dt). You're essentially ensuring that
                    # the datapoints fall within the last-x-day time period by checking that they intersect.
                    df_slice = df.loc[(end_dt > df[ts_field]) & (start_dt <= cutoff_series), :]
                    df_slice['agg_proportion'] = df_slice.apply(
                        lambda r: min(
                            1.0,
                            (min(end_dt, r[ts_cutoff_field]) - max(r[ts_field], start_dt)).total_seconds() /
                            (coalesce([r[ts_cutoff_field], end_dt])-r[ts_field]).total_seconds()
                            ),
                        axis=1)
                    df_slice[fld] = df_slice.apply(lambda r: r[fld] * r['agg_proportion'], axis=1)
                elif aggregate in ('last', 'first'):
                    df_slice = df.loc[(end_dt > df[ts_field]), :]
                else:
                    df_slice = df.loc[(end_dt > df[ts_field]) & (start_dt <= df[ts_field]), :]

                new_df[field_name] = df_slice.groupby(by=group_field)[fld].aggregate(aggregate).fillna(0)

        if joindate_field:
            join_df = df.loc[~df[joindate_field].isnull(), :]
            join_df[joindate_field] = pd.to_datetime(join_df[joindate_field])
            join_df[ts_field] = pd.to_datetime(join_df[ts_field])

            post_join_records = (join_df[ts_field] >= join_df[joindate_field])
            if presignup:
                pre_join_records = ~post_join_records
                new_df[f'presignup_{fld}{suffix}'] = join_df.loc[(pre_join_records), :]\
                    .groupby(by=group_field)[fld].aggregate(aggregate).fillna(0)
            if postsignup:
                if aggregate in ('last', 'first'):
                    new_df[f'postsignup_{fld}{suffix}'] = join_df\
                        .groupby(by=group_field)[fld].aggregate(aggregate).fillna(0)
                else:
                    new_df[f'postsignup_{fld}{suffix}'] = join_df.loc[(post_join_records), :]\
                        .groupby(by=group_field)[fld].aggregate(aggregate).fillna(0)

            if firstxdays:
                time_periods = dict()
                if firstxdays is True:
                    firstxdays = [30, 60, 90]
                for days in firstxdays:
                    time_periods[f'first{days}day_{fld}{suffix}'] = (0, days)
                    if include_preceding_period:
                        time_periods[f'first{days}day_following{days}_{fld}{suffix}'] = (days, days*2)

                for field_name, (start_days, end_days) in time_periods.items():
                    join_df['end_dt'] = join_df[joindate_field] + timedelta(days=end_days)
                    join_df['start_dt'] = join_df[joindate_field] + timedelta(days=start_days)
                    if aggregate in ('last', 'first'):
                        df_slice = join_df.loc[(join_df[ts_field] <= join_df['end_dt'])
                                & (join_df[ts_field] > join_df['start_dt']), :]
                    else:
                        df_slice = join_df.loc[(post_join_records) & (join_df[ts_field] <= join_df['end_dt'])
                                & (join_df[ts_field] > join_df['start_dt']), :]

                    if ts_cutoff_field:
                        df_slice['agg_proportion'] = df_slice.apply(
                            lambda r: min(
                                1.0,
                                (min(r['end_dt'], r[ts_cutoff_field]) - max(r[ts_field], r['start_dt'])).total_seconds() /
                                (coalesce([r[ts_cutoff_field], curr_dt]) - r[ts_field]).total_seconds()
                                if (coalesce([r[ts_cutoff_field], curr_dt]) - r[ts_field]).total_seconds() > 0 else 0),
                            axis=1)
                        df_slice[fld] = df_slice.apply(lambda r: r[fld] * r['agg_proportion'], axis=1)
                    new_df[field_name] = df_slice.groupby(by=group_field)[fld].aggregate(aggregate).fillna(0)

    if retain_fields:
        retained = df.set_index(group_field).loc[:, retain_fields].drop_duplicates()
        new_df = new_df.merge(retained, left_index=True, right_index=True, how='left')
    return new_df


def get_refresh_date_key(file_addr):
    file_name = os.path.basename(file_addr)
    refresh_string_suffix = '_last_modified'
    return file_name + refresh_string_suffix


def refresh_required(min_date: dt.datetime, file_addr: str, session_info: [dict, SqliteDict] = {}) -> bool:

    res = False
    last_modified_key = get_refresh_date_key(file_addr)

    if last_modified_key in session_info:
        if session_info[last_modified_key] < min_date:
            res = True

    elif os.path.exists(file_addr):
        file_last_mod_dt = convert_to_ts(os.stat(file_addr).st_mtime)
        if file_last_mod_dt < min_date:
            res = True
        session_info[last_modified_key] = file_last_mod_dt

    else:  # If the file doesn't exist AND the session_info argument wasn't passed, then the data must be refreshed
        res = True

    print('Refresh{0} required for data in file {1}'.format(' IS' if res else ' NOT', os.path.basename(file_addr)))
    return res


def wait_until_biz_hours():
    now = dt.datetime.now()
    end_biz_hours = now.replace(hour=20, minute=0, second=0, microsecond=0)
    start_biz_hours = now.replace(hour=8, minute=0, second=0, microsecond=0)
    while True:
        now = dt.datetime.now()
        if start_biz_hours <= now < end_biz_hours:
            time_remaining = timer_hms(end_biz_hours - dt.datetime.now())
            print(f'\rIt is currently business hours in the USA (8AM-8PM EST). '
                  f'{time_remaining} remaining until we can run the query', end='')
            time.sleep(1)
        else:
            end_biz_hours = now.replace(hour=20, minute=0, second=0, microsecond=0)
            start_biz_hours = now.replace(hour=8, minute=0, second=0, microsecond=0)
            if not (start_biz_hours <= now < end_biz_hours):
                print('\nOutside biz hours! Ready to go!')
                break
    return



def dedupe_and_check_unique(df: pd.DataFrame, fields: Union[tuple, list]=tuple()):
    df.drop_duplicates(inplace=True)
    for fld in fields:
        if df[fld].shape != df[fld].unique().shape:
            err_str = f'Non-unique field [{fld}] found in DataFrame!'
            print(err_str)
            print(df[fld].value_counts(sort=True, ascending=False))
            raise ValueError(err_str)


def get_financial_metrics(df: pd.DataFrame, prefix: str):
    cost = '_est_total_cost'
    margin = '_est_total_margin'
    margin_pct = '_est_total_margin_pct'
    csm = '_natero_est_csm_cost'
    css = '_est_css_cost'
    ser = '_est_server_cost'
    rev = '_revenue'
    margin_neg = '_est_negative_margin_flag'

    if all(fld in df.columns for fld in (f'{prefix}{rev}',f'{prefix}{css}', f'{prefix}{csm}', f'{prefix}{ser}')):
        df[f'{prefix}{cost}'] = df[f'{prefix}{css}'].fillna(0.0) + \
                                df[f'{prefix}{csm}'].fillna(0.0) + \
                                df[f'{prefix}{ser}'].fillna(0.0)
        df[f'{prefix}{margin}'] = df[f'{prefix}{rev}'] - df[f'{prefix}{cost}']
        df[f'{prefix}{margin_pct}'] = df[f'{prefix}{margin}'] / df[f'{prefix}{rev}']
        df[f'{prefix}{margin_neg}'] = df[f'{prefix}{margin}'] < 0

    return df


def coalesce(vals: Union[list, tuple], addl_null_vals: Union[list, tuple]=tuple()):
    for v in vals:
        if not (pd.isna(v) or v in addl_null_vals):
            return v
    else:
        return None


def merge_and_drop(df1: pd.DataFrame, df2: pd.DataFrame, **kwargs):
    df1 = pd.merge(df1, df2, suffixes=('', '_drop'), **kwargs)
    drop_cols = [c for c in df1.columns.tolist() if '_drop' in c]
    return df1.drop(columns=drop_cols, axis=1)


def get_hscout_token() -> str:
    with requests.session() as s:
        res = s.post('https://api.helpscout.net/v2/oauth2/token',
                     data={'grant_type': 'client_credentials',
                           'client_id': f'{os.environ["HELPSCOUT_APP_ID"]}',
                           'client_secret': f'{os.environ["HELPSCOUT_APP_SECRET"]}'
                           }
                     ).json()['access_token']
    return res


def parse_mau(plan_desc: str) -> Union[int, None]:
    plan_desc = plan_desc.lower()
    if 'hobby' in plan_desc:
        return 250
    elif 'bootstrap' in plan_desc:
        return 1000
    elif 'startup' in plan_desc:
        return 1000
    elif 'growth' in plan_desc:
        return 20000
    elif '999permonth' in plan_desc:
        return 25000
    elif 'business' in plan_desc:
        return 50000
    else:
        try:
            mau = re.search('-\s([,0-9]+)\smau.*', plan_desc).group(1)
            return int(mau.replace(',',''))
        except (IndexError, AttributeError):
            return None


def to_pickle(file_addr: str, obj: object, print_result: bool=True):
    with open(file_addr, mode='wb') as file_obj:
        pickle.dump(obj, file_obj)
        if print_result:
            print(f'Dumped data to Pickle file: {file_addr}')


def from_pickle(file_addr: str, print_result: bool=True) -> object:
    with open(file_addr, mode='rb') as file_obj:
        obj = pickle.load(file_obj)
        if print_result:
            print(f'Loaded data from Pickle file: {file_addr}')
        return obj


def open_shelf(file_name: str, new: bool = False, raise_errors: bool = True, autocommit: bool = True) -> SqliteDict:
    try:
        sld = SqliteDict(file_name, autocommit=autocommit)
        if new:
            print('Deleting old shelf file and creating a new one.')
            sld.clear()
    except Exception as e:
        print(e)
        if raise_errors:
            raise e
        print('Error during shelf open action. Since raise_errors is false, starting from a new file.')
        sld = open_shelf(file_name, new=True, raise_errors=True)
    return sld


# Modify a file by renaming it and copying it to a new location,
# in this case, the //data folder in the base directory for this report
def convert_file(orig_file, new_file, encoding):

    csv.field_size_limit(sys.maxsize)
    if orig_file == new_file:
        return orig_file
    print_and_log("Renaming: [%s] to [%s]" % (orig_file, new_file), 'info')
    retries = 3
    while True:
        try:
            if os.path.exists(new_file):
                os.remove(new_file)
            break
        except PermissionError:
            if retries > 0:
                retries -= 1
                time.sleep(3)
                pass
            else:
                print_and_log("Unable to delete existing file")
                raise
    try:
        with open(orig_file, mode='r', encoding=encoding) as file:
            dialect = csv.Sniffer().sniff(file.readline())
            dialect.doublequote = True
            data_iter = csv.reader(file, dialect=dialect)
            file.seek(0)
            if encoding.upper().replace('-', '').replace('_', '') in ('UTF8', 'LATIN1', 'CP1252'):
                dialect = csv.excel
            else:
                dialect = csv.excel_tab
            dialect.quoting = csv.QUOTE_MINIMAL
            dialect.doublequote = True
            # dialect.delimiter = '\t'
            dialect.quotechar = '"'
            dialect.lineterminator = '\n'
            with open(new_file, mode='w', encoding=encoding) as file2:
                data_writer = csv.writer(file2, dialect=dialect)
                data_writer.writerows(data_iter)

        # shutil.move(orig_file, new_file)
        # print("Moved to: %s" % download_dir)
    # except PermissionError:
    #     shutil.copy2(orig_file, new_file)
    #     print("Copied to: %s" % download_dir)
    #     pass
    except FileNotFoundError as e:
        print_and_log("shutil.move failed. WTF?!?!?")
        print_and_log(e)

    return new_file


def create_email_message(sender: str, to: list, subject: str, message_text: str, file):
    """Create a message for an email.

    Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    file: The path to the file to be attached.

    Returns:
    An object containing a base64url encoded email object.
    """

    from os.path import basename
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage
    from email.mime.base import MIMEBase
    from email.mime.audio import MIMEAudio
    import mimetypes

    message = MIMEMultipart()
    message['to'] = ", ".join(to)
    message['from'] = sender
    message['subject'] = subject

    msg = MIMEText(message_text)
    message.attach(msg)

    if file:
        content_type, encoding = mimetypes.guess_type(file)

        if content_type is None or encoding is not None:
            content_type = 'application/octet-stream'
        main_type, sub_type = content_type.split('/', 1)

        if main_type == 'text':
            fp = open(file, 'rt')
            msg = MIMEText(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'image':
            fp = open(file, 'rb')
            msg = MIMEImage(fp.read(), _subtype=sub_type)
            fp.close()
        elif main_type == 'audio':
            fp = open(file, 'rb')
            msg = MIMEAudio(fp.read(), _subtype=sub_type)
            fp.close()
        else:
            fp = open(file, 'rb')
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
            fp.close()

        filename = basename(file)
        msg.add_header('Content-Disposition', 'attachment', filename=filename)
        message.attach(msg)

    return message.as_string()


def get_arg_response(arg: str, opt: str):
    if arg in ('f', 'false'):
        return False
    elif arg in ('t', 'true'):
        return True
    else:
        raise ValueError('Wrong argument given [%s] for option %s' % (arg, opt))


class ArgDef:
    def __init__(self, long_form: str, short_form: str, default_val, possible_vals: list):
        self.long_form = long_form
        self.short_form = short_form
        self.default_val = default_val
        self.possible_vals = possible_vals

    def __iter__(self) -> tuple:
        return self.long_form, self.short_form, self.default_val, self.possible_vals


def process_args(args: Union[tuple, list], arglist: Union[tuple, list]) -> list:
    if len(args) > 1:
        if args[1:][0] == '-h':
            print("You may call this file with any of the following arguments:")
            for r in arglist:
                key, abbr, default, possible_vals = r
                if callable(possible_vals):
                    possible_vals = "".join(inspect.getsourcelines(possible_vals)[0]).split(',')[
                        -1].strip().rstrip(')')

                print('"--{0}"\n\tAbbreviation: "-{1}"\n\tDefault: "{2}"\n\tPossible Values: {3}'
                      .format(key, abbr, default
                              , "".join(inspect.getsourcelines(possible_vals)[0]).split(',')[-1].strip()[:-1]
                              if callable(possible_vals) else possible_vals))
            sys.exit(2)

    str_short_args = 'h' + ''.join(['%s' % r[1] + ':' for r in arglist])
    list_long_args = ['%s=' % r[0] for r in arglist]
    try:
        print("Arguments:", args)
        opts, args = getopt.getopt(args[1:], str_short_args, list_long_args)
        print("User-provided arguments: ", opts)
    except getopt.GetoptError:
        print('Incorrect/unparsable arguments given. Check your command line argument.')
        sys.exit(2)

    vals = dict()
    for r in arglist:
        key, abbr, default, possible_vals = r
        val = default
        for opt, arg in opts:
            # if not arg:
            #     arg = args[0]
            #     del args[0]
            arg = arg.lower()
            if opt in ("-%s" % abbr, "--%s" % key):
                if isinstance(default, str):
                    val = arg
                elif isinstance(default, bool):
                    val = get_arg_response(arg, opt)
                elif isinstance(default, (float, int)):
                    val = float(arg)
                break

        if not (isinstance(possible_vals, (list, tuple)) or callable(possible_vals)):
            possible_vals = [possible_vals]

        if (not possible_vals) \
                or (callable(possible_vals)
                    and possible_vals(val)) \
                or (isinstance(possible_vals, (list, tuple))
                    and (val in possible_vals)):
            vals[key] = val
            print('%s:\t' % key, val)
        else:
            print("Invalid value [{0}] provided. Must conform to the following: {1}".format(val, possible_vals))
            sys.exit(1)

    return vals.items()


@contextmanager
def csv_iterator(csv_file, encoding: str = None):
    if not encoding:
        encoding = get_encoding_type(csv_file)
    file = open(csv_file, mode='r', encoding=encoding)
    dialect = csv.Sniffer().sniff(file.readline())
    dialect.doublequote = True
    data_iter = csv.reader(file, dialect=dialect)
    file.seek(0)
    try:
        yield data_iter
    finally:
        file.close()


class TabCmd(object):
    def __init__(self, tabcmd: str, server: str, username: str, linux: bool = False):
        self.tabcmd = tabcmd
        self.server = server
        self.username = username
        self.linux = linux is True

        # resp = 5
        # dir = self.tabcmd
        # while resp == 5:
        #     dir = '\\'.join(dir.split('\\')[:-1])
        #     resp = control_dir(dir=dir)
        # lol=1
        # if os.path.isfile(tabcmd):
        #     import stat
        #     os.chmod(tabcmd, stat.S_IEXEC)

    def _execute_command(self, command):
        # In Tableau 9.2 we have to append --no-certcheck for linux connections
        # More info here: https://community.tableau.com/message/450517#450517
        if self.linux and '--no-certcheck' not in command:
            command += ["--no-certcheck"]

        if isinstance(command, list):
            command = ' '.join(command)
        print(command)
        subprocess.run(command)

    def login(self, site, password=None, certcheck: bool = False):

        command = [
            self.tabcmd,
            'login',
            "-u", "%s" % self.username,
            "-p", "%s" % password,
            "-s", "%s" % self.server,
            "-t", "%s" % site,
        ]

        if not certcheck:
            command += ['--no-certcheck']

        self._execute_command(command)

    def create_project(self, name: str, description: str = None):
        command = [
            self.tabcmd,
            'createproject',
            "-n", "%s" % name,
        ]
        if description is not None:
            command += ["-d", description]
        self._execute_command(command)

    def delete_project(self, name: str):
        command = [
            self.tabcmd,
            'deleteproject',
            name
        ]
        self._execute_command(command)

    def refresh_extracts(self, name: str, synchronous: bool = True):
        command = [
            self.tabcmd,
            'refresh_extracts'
        ]
        if synchronous:
            command += ['--synchronous']
        self._execute_command(command)

    def export(self, url: str, output_path: str, refresh: bool = True, certcheck: bool = False,
               width: int = 0, height: int = 0):

        if '?' not in url:
            url += '?'
        elif url[-1] != '?':
            url += '&'

        url += ':refresh=' + ('yes' if refresh else 'no')

        tabcmd_addr = '"{0}"'.format(self.tabcmd)
        url = '"{0}"'.format(url)
        command = [tabcmd_addr, 'export', url]

        if '.pdf' in output_path:
            command += ['--pdf']
        elif '.png' in output_path:
            command += ['--png']

        if width > 0:
            command += ['--width ' + str(width)]

        if height > 0:
            command += ['--height ' + str(height)]

        command += ['--filename "{0}"'.format(output_path)]

        if not certcheck:
            command += ['--no-certcheck']

        self._execute_command(command)

    def get(self, url: str, output_path: str, refresh: bool = True, certcheck: bool = False, size: str = ''):

        if '?' not in url:
            url += '?'
        elif url[-1] != '?':
            url += '&'

        url += ':refresh=' + ('yes' if refresh else 'no')

        if size:
            url += '&:size=' + size

        tabcmd_addr = '"{0}"'.format(self.tabcmd)
        url = '"{0}"'.format(url)
        command = [tabcmd_addr, 'get', url]

        command += ['--filename "{0}"'.format(output_path)]

        if not certcheck:
            command += ['--no-certcheck']

        self._execute_command(command)



def impute_if_any_nulls(impute_df: pd.DataFrame, verbose: bool=False):
    from fancyimpute import BiScaler, NuclearNormMinimization, MatrixFactorization, IterativeSVD
    impute_names = impute_df.columns.values.tolist()
    impute_index = impute_df.index.values
    for imputer in [BiScaler, NuclearNormMinimization, MatrixFactorization, IterativeSVD]:
        if impute_df.isnull().any().any():
            print(f'Imputation: Null values are in the DF. Running imputation using "{imputer.__name__}"')
            impute_df = imputer(verbose=verbose).fit_transform(impute_df.values)
            impute_df = pd.DataFrame(data=impute_df, columns=impute_names, index=impute_index)
        else:
            break
    # else:
    #     print('Imputation: Unable to eliminate all NULL values from the dataframe! FIX THIS!')

    for n in impute_names.copy():
        if impute_df[n].isnull().any().any():
            print('Field [{0}] was still empty after imputation! Removing it!'.format(n))
            impute_names.remove(n)

    return impute_df, impute_names


# if __name__ == '__main__':
#     load_env()
#     t = get_hscout_token()
#     print(t)


if __name__ == '__main__':
    d = json.loads(open('tags.json').read())
    print(d['tags'])
    df = pd.DataFrame(d['tags'])
    df.to_csv('tags.csv')
