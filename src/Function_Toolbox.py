import logging
import Settings
from chardet.universaldetector import UniversalDetector
from contextlib import contextmanager
import pickle
import os
import csv
import pandas as pd
import numpy as np
import getopt
import sys
import inspect
import shutil
import math
import time
import tensorflow as tf
from datetime import date
import platform
import subprocess
import importlib
from re import search as re_search
from fancyimpute import BiScaler, NuclearNormMinimization, MatrixFactorization, IterativeSVD

global log


def create_logger(report_name: str):
    s = Settings.Settings(report_name=report_name)
    logfile_addr = s.get_logfile_addr()
    with open(logfile_addr, 'w'):  # Clears the logfile
        pass

    global log
    # create a logger
    try:
        log = logging.getLogger(get_report_name())
    except:
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
            raise Exception()
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
    return importlib.import_module(get_exec_file()).__report_name__


def get_report_owner() -> str:
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


def reverse_enumerate(l):
    for index in reversed(range(len(l))):
        yield index, l[index]


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


def skflow_rnn_input_fn(x: tf.Tensor) -> list:
    return tf.split(x, x.shape[1].value, 1)

def str_as_header(txt: str, char: str='=', end_chars: int=5) -> str:
    end_str = char * (end_chars*1 + 1)
    txt = end_str + ' ' + txt + ' ' + end_str
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


def prepend_prefix(prefix: str, name: str) -> str:
    if isinstance(prefix, str) \
            and not prefix == name[:len(prefix)]:
        name = prefix + name
    return name


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


def get_arg_response(arg: str, opt: str):
    if arg in ('f', 'false'):
        return False
    elif arg in ('t', 'true'):
        return True
    else:
        raise ValueError('Wrong argument given [%s] for option %s' % (arg, opt))


def impute_if_any_nulls(impute_df: pd.DataFrame, verbose: bool=False):
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


def process_args(args, arglist):
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


class XML_Query:
    def __init__(self, prefix: str,
                 tags_to_traverse: list,
                 tags_to_avoid: list = [],
                 tags_to_skip: list = [],
                 folder: str = None,
                 specific_tags: list = [],
                 skip_all_child_nodes: bool = False,
                 ):

        from DataDelivery import XML_Parser

        self.prefix = prefix
        self.df_dict = dict()
        self.tags_to_traverse = tags_to_traverse
        self.tags_to_avoid = tags_to_avoid
        self.tags_to_skip = tags_to_skip
        self.specific_tags = specific_tags
        self.skip_all_child_nodes = skip_all_child_nodes
        self.dialect = csv.excel
        self.dialect.lineterminator = '\n'
        if not folder:
            try:
                s = Settings.Settings(get_report_name())
                folder = s.get_report_data_dir()
            except AttributeError as e:
                s = Settings.Settings('default')
                folder = s.get_default_data_dir()
        self.pickle_file = folder + prefix + '.p'
        self.columns = list()
        self.parser = XML_Parser.Parser()

    def add_df(self, sn: str, df: pd.DataFrame):
        self.df_dict[sn] = df

    def add_file_addr(self, file_addr: str):
        self.file_addr = file_addr
        self.first_pickle_write = True
        os.makedirs(os.path.dirname(self.file_addr), exist_ok=True)

    def set_pickle(self, df: pd.DataFrame, reset_cols: bool = False):
        with open(self.pickle_file, mode='wb') as f:
            pickle.dump(df, f)
            self.first_pickle_write = True
            if reset_cols:
                self.columns = df.columns.values

    def add_to_pickle(self, df: pd.DataFrame, mode: str = 'truncate'):

        [self.columns.append(c) for c in list(df.columns) if c not in self.columns]
        if mode == 'truncate' and self.first_pickle_write:
            with open(self.pickle_file, mode='wb') as f:
                pickle.dump(df, f)
                self.first_pickle_write = False
        elif mode == 'append' or (mode == 'truncate' and not self.first_pickle_write):
            with open(self.pickle_file, mode='ab') as f:
                pickle.dump(df, f)
        else:
            raise ValueError('Unsure what to do with the current Dataframe. Check the code or your mode setting.')

    def to_df(self, root):
        df = self.parser.xmlToDF(root, self.tags_to_traverse, self.tags_to_avoid, self.tags_to_skip,
                                 self.specific_tags, self.skip_all_child_nodes)
        return df

    def get_df(self, remove_fields: list = None, delete_after: bool = True):
        first_write = True
        df = pd.DataFrame()

        if not self.columns:
            self.columns = set()
            for df in self.pickle_loader():
                self.columns.update(df.columns)
            self.columns = list(self.columns)

        try:
            for qty, temp_df in enumerate(self.pickle_loader()):
                # df_cols = list(temp_df.columns)
                # for c in self.columns:
                #     if c not in df_cols:
                #         df[c] = np.nan
                # temp_df = temp_df[self.columns]
                # if first_write:
                #     df = temp_df
                #     first_write = False
                # else:
                #     df = df.append(temp_df, ignore_index=True)

                if first_write:
                    df = temp_df
                    first_write = False
                else:
                    df = pd.concat([df, temp_df], axis=0, ignore_index=True)

        except (pickle.UnpicklingError):
            print('[%s] Unpickling Error. Number of objects unpickled: %s' % (self.prefix, qty))
            pass

        except (FileNotFoundError):
            print('[{0}] No data found for the XML query constraints'.format(self.prefix))
            pass

        # df = df[self.columns]
        if delete_after:
            try:
                os.remove(self.pickle_file)
            except FileNotFoundError:
                pass

        if remove_fields:
            df.drop(remove_fields, errors='ignore', inplace=True, axis=1)
        return df

    def to_csv(self, mode: str = 'truncate', remove_fields: list = None, delete_after: bool = True):
        first_write = True
        try:
            for qty, df in enumerate(self.pickle_loader()):
                if remove_fields:
                    df.drop(remove_fields, errors='ignore', inplace=True, axis=1)
                df_cols = list(df.columns)
                for c in self.columns:
                    if c not in df_cols:
                        df[c] = np.nan
                df = df[self.columns]
                if mode == 'truncate' and first_write:
                    df.to_csv(self.file_addr, header=True, mode='w', index=False, encoding='utf_8')
                    first_write = False
                elif mode == 'append' or (mode == 'truncate' and not first_write):
                    df.to_csv(self.file_addr, header=False, mode='a', index=False, encoding='utf_8')

        except (pickle.UnpicklingError):
            print('[%s] Unpickling Error. Number of objects unpickled: %s' % (self.prefix, qty))
            pass
        except (FileNotFoundError):
            print('[{0}] No data found for the XML query constraints'.format(self.prefix))
            pass

        if delete_after:
            try:
                os.remove(self.pickle_file)
            except FileNotFoundError:
                pass

    def get_csv(self):
        return pd.read_csv(self.file_addr, index_col=False, encoding='utf_8')

    def pickle_loader(self):
        num_objs = 0
        try:
            with open(self.pickle_file, mode='rb') as f:
                while True:
                    yield pickle.load(f)
                    num_objs += 1
        except EOFError:
            print('[%s] Number of objects unpickled: %s' % (self.prefix, num_objs))
            pass