import os, tempfile
import csv
from collections import deque
from pandas import read_excel, DataFrame

# set global variables
default_download_dir = os.path.expanduser('~/Downloads')
os.makedirs(default_download_dir, exist_ok=True)

class query:

    def __init__(self, file_addr: str):

        self.file_addr = file_addr
        self.tempDir = tempfile.mkdtemp()

    def goToQuery(self, sheet_name: str):

        self.sheet_name = sheet_name

    def run_query(self):

        encoding = 'utf_8'
        df = read_excel(self.file_addr, self.sheet_name)
        file_name = tempfile.mkdtemp() + '\\XLS_Query.csv'
        print('Dumping the XLS query results to a file...')
        df.to_csv(file_name, encoding=encoding, index=False, quoting=csv.QUOTE_MINIMAL)
        print("\rXLS query CSV file rows written: %s" % (len(df.index)))

        return None, file_name, encoding

    def close_driver(self):
        pass
