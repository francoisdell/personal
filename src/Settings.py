import os
import logging
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

class Settings:
    def __init__(self, report_name: str):
        self.report_name = report_name
        self.root_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).replace('\\', '/').split('/')[:-1])
        self.root_query_dir = self.root_dir + '/Queries'
        self.root_data_dir = self.root_dir + '/Data Files'
        self.default_base_dir = self.root_dir + '/Analytics Reports'
        self.default_backup_dir = self.root_dir + '/Backup Files'
        self.root_logfile_dir = self.root_dir + '/logfiles'
        self.root_model_dir = self.root_dir + '/Models'  # Will be appended to the end of the user-defined base directory for the project
        self.default_data_dir = '/Data'  # Will be appended to the end of the user-defined base directory for the project
        self.default_vba_dir = '/VBA'  # Will be appended to the end of the user-defined base directory for the project
        self.default_query_dir = '/Queries'  # Will be appended to the end of the user-defined base directory for the project
        self.default_model_dir = 'Models'  # Will be appended to the end of the user-defined base directory for the project
        self.default_query_wait_time = 5400  # 3600 = 1 hour
        self.default_max_threads = 5  # Max simultaneous query threads that can be run at once
        self.default_backup_period_days = 14  # Number of days to retain backup files before deleting
        self.default_loading_db = 'VMAXAnalytics'
        self.default_vba_use_whole_directory = False
        self.default_is_test = False
        self.default_ignore_if_exists = 0
        self.default_ignore_if_queried_within_hrs = 11

    def get_query_dir(self) -> str:
        return self.get_dir(self.root_query_dir + '/' + self.report_name)

    def get_default_data_dir(self) -> str:
        return self.get_dir(self.root_data_dir + '/' + self.report_name)

    def get_report_data_dir(self) -> str:
        return self.get_dir(self.root_data_dir + '/' + self.report_name)

    def get_model_dir(self) -> str:
        return self.get_dir(self.root_model_dir + '/' + self.report_name)

    def get_backup_dir(self) -> str:
        return self.get_dir(self.default_backup_dir + '/' + self.report_name)

    def get_reports_dir(self) -> str:
        return self.get_dir(self.default_base_dir + '/' + self.report_name)

    def get_vba_dir(self) -> str:
        return self.get_dir(self.get_reports_dir() + self.default_vba_dir)

    def get_logfile_addr(self) -> str:
        logfile_dir = self.get_dir(self.root_logfile_dir)
        return logfile_dir + '/logfile_{0}.log'.format(self.report_name)

    def get_dir(self, dir: str) -> str:
        os.makedirs(dir, exist_ok=True)
        return dir
