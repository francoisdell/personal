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
import time
from datetime import date
import platform
import subprocess
from DataDelivery import XML_Parser

global log

def create_logger(report_name: str):

    s = Settings.Settings(report_name=report_name)
    logfile_addr = s.get_logfile_addr()
    with open(logfile_addr, 'w'): # Clears the logfile
        pass

    # create a logger
    logger = logging.getLogger(getExecFile())
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(logfile_addr)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    global log
    log = logger

def get_logger():
    global log
    log = logging.getLogger(getExecFile())

def printAndLog(msg, log_type: str=None):
    print(msg)
    try:
        log
    except:
        get_logger()

    if log_type == 'info':
        log.info(msg)
    elif log_type == 'exception':
        log.exception(msg)
    elif log_type == 'error':
        log.error(msg)
        
def getExecFile():
    # Obtains the name of the python file which was originally executed
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__).replace('\\', '/')
    except:
        sFile = sys.executable
    sFile = os.path.splitext(sFile.split('/')[-1])[0]

    return sFile

def purgeDir(path, days):
    printAndLog("Scanning %s for files older than %s days." % (path, days), 'info')
    for f in os.listdir(path):

        now = time.time()
        filepath = os.path.join(path, f)
        last_modified = os.stat(filepath).st_mtime
        if last_modified < now - (days * 86400): # 86400 = number of seconds in one day
            printAndLog("Removing backup older than %s days: %s" % (f, days), 'info')
            try:
                shutil.rmtree(filepath)
                msg = 'DELETED'
            except:
                msg = 'FAILED TO DELETE'
            print('{0}: {1} (Last modified on: {2})'.format(msg, f, date.fromtimestamp(last_modified)))

def normalizeCSV(csvFile, new_csvFile, encoding: str = None):
    with open(csvFile, mode='r', encoding=encoding) as file:
        dialect = csv.Sniffer().sniff(file.readline())
        dialect.doublequote = True
        data_iter = csv.reader(file, dialect=dialect)
        file.seek(0)
        dialect.quoting = csv.QUOTE_MINIMAL
        dialect.delimiter = ','
        dialect.quotechar = '"'
        dialect.lineterminator = '\n'
        with open(new_csvFile, mode='w', encoding=encoding) as file2:
            data_writer = csv.writer(file2, dialect=dialect)
            data_writer.writerows(data_iter)

def control_dir(dir: str):
    if "Windows" in platform.architecture()[1]:
        own_cmd = 'takeown /F "%s" /R /d Y' % dir.replace('/', '\\')
        control_cmd = 'icacls "%s" /grant Administrator:(OI)(CI)F /T /Q' % dir.replace('/', '\\')
    else:  # Add something here for Unix/Linux?!?!?
        own_cmd = None
        control_cmd = None

    if own_cmd:
        print("Ownership Command: %s" % own_cmd)
        subprocess.call(own_cmd)

    if control_cmd:
        print("Control Command: %s" % own_cmd)
        subprocess.call(control_cmd)

def get_encoding_type(current_file: str):
    print("Determining file encoding: [%s]" % current_file, 'info')
    detector = UniversalDetector()
    detector.reset()
    for line in open(current_file, 'rb'):
        detector.feed(line)
        if detector.done: break
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

    str_short_args = 'h' + ':'.join(['%s' % r[1] for r in arglist])
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
            arg = arg.lower()
            if opt in ("-%s" % abbr, "--%s" % key):
                if isinstance(default, str):
                    val = arg
                elif isinstance(default, bool):
                    val = get_arg_response(arg, opt)
                elif isinstance(default, float):
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
def getCSVIter(csvFile, encoding: str = None):
    if not encoding:
        encoding = get_encoding_type(csvFile)
    file = open(csvFile, mode='r', encoding=encoding)
    dialect = csv.Sniffer().sniff(file.readline())
    dialect.doublequote = True
    data_iter = csv.reader(file, dialect=dialect)
    file.seek(0)
    try:
        yield data_iter
    finally:
        file.close()

class XML_Query:
    def __init__(self, prefix: str, tags_to_traverse: list, tags_to_avoid: list=[], tags_to_skip: list=[], folder: str=None):
        self.prefix = prefix
        self.df_dict = dict()
        self.tags_to_traverse = tags_to_traverse
        self.tags_to_avoid = tags_to_avoid
        self.tags_to_skip = tags_to_skip
        self.dialect = csv.excel
        self.dialect.lineterminator = '\n'
        self.pickle_file = (folder + '/' if folder else '') + prefix + '.p'
        self.columns = list()
        self.parser = XML_Parser.Parser()

    def add_df(self, sn: str, df: pd.DataFrame):
        self.df_dict[sn] = df
    
    def add_file_addr(self, file_addr: str):
        self.file_addr = file_addr
        self.first_pickle_write = True
        os.makedirs(os.path.dirname(self.file_addr), exist_ok=True)
    
    def add_to_pickle(self, df: pd.DataFrame, mode: str='truncate'):
    
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
        df = self.parser.xmlToDF(root, self.tags_to_traverse, self.tags_to_avoid, self.tags_to_skip)
        return df

    def to_csv(self, mode: str='truncate', remove_fields: list=None):
        first_write = True
        num_objs = 0
        try:
            with open(self.pickle_file, mode='rb') as f:
                while True:
                    df = pickle.load(f)
                    if remove_fields:
                        df.drop(remove_fields, errors='ignore', inplace=True, axis=1)
                    num_objs += 1
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
    
        except (EOFError, pickle.UnpicklingError):
            print('[%s] Number of objects unpickled: %s' % (self.prefix, num_objs))
            os.remove(self.pickle_file)
            pass
        except (FileNotFoundError):
            print('[{0}] No data found for the XML query constraints (num_objs = {1})'.format(self.prefix, num_objs))
            pass
    
    def get_csv(self):
        return pd.read_csv(self.file_addr, index_col=False, encoding='utf_8')
    
    def pickle_loader(self):
        try:
            while True:
                yield pickle.load(self.pickle_file)
        except EOFError:
            pass