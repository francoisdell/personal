__author__ = 'mackea1'
import numpy as np
import re
import tempfile, csv
from collections import deque
import codecs
import pymssql

class query:

    def __init__(self, query_dir: str, db_name: str):

        self.default_encoding = 'utf_8'
        self.query_dir = query_dir

        if db_name == 'SYR':
            server = 'DBABCV30SQL1\\SYRPRODBCV'
            server = 'SYR-BCV'
            database = 'SYR'
            uid = 'ICWebUser'
            pwd = '4SYR_bcv'

        elif db_name == 'ElcidStaging':
            server = 'FRER6-PROD-ETL.corp.emc.com'
            server = 'PROD-ETL'
            database = 'ElcidStaging'
            uid = 'svc_mosssymmeng'
            pwd = 'sadAcuK3'

        elif db_name == 'OPT':
            server = 'ESDBISSQLPRD01'
            server = 'OPT'
            database = 'MCC_Reporting'
            uid = 'VmaxTCEIQMUserAcct'
            pwd = 'vM@xtce1qMu$er@cct@176ss'

        elif db_name == 'DUDL':
            server = 'DASSQLCLD055'
            server = 'DUDL'
            database = 'dudl'
            uid = 'mosssymmeng'
            pwd = 'Welcome@123'

        else:
            print('Unknown database specified! Failed!')

        import sqlalchemy
        # engine = sqlalchemy.create_engine("mssql+mymssql://%s:%s@%s/%s" % (uid, pwd, server, database), encoding='latin1', convert_unicode=True)
        engine = sqlalchemy.create_engine("mssql+pyodbc://%s:%s@%s" % (uid, pwd, server)
                                          , encoding=self.default_encoding
                                          , convert_unicode=True)
        self.conn = engine.raw_connection()
        self.cursor = self.conn.cursor()

        # import pypyodbc
        # print("Connecting to MSSQL database...", end='')
        # self.conn = pypyodbc.connect('DRIVER={SQL Server Native Client 11.0};'
        #                            'SERVER=%s;'
        #                            'DATABASE=%s;'
        #                            'UID=%s;'
        #                            'PWD=%s' % (server, database, uid, pwd))
        # self.cursor = self.conn.cursor()


    def goToQuery(self, query_file):

        # Commit the download directory
        with open("%s\\%s" % (self.query_dir, query_file), "r") as myfile:
            self.query_text = myfile.read().replace('ï»¿','')


    def runQuery(self):

        possible_encodings = deque(['ascii', 'utf_32', 'utf_16_le', 'utf_16', 'cp1252', 'latin_1', 'utf_8'])
        try:
            possible_encodings.remove(self.default_encoding)
            possible_encodings.append(self.default_encoding)
        except:
            pass
        file_name = tempfile.mkdtemp() + '\\SYR_Query.csv'
        encoding = None
        row_count = 0
        row_iterator_size = 100000
        while True:
            print('Executing the MSSQL query...')
            self.cursor.execute(self.query_text)
            print('Dumping the MSSQL query results to a file...')
            try:
                encoding = possible_encodings.pop()
                with open(file_name, "w", newline='\n', encoding=encoding) as f:
                    writer = csv.writer(f, dialect=csv.excel)
                    writer.writerow(i[0] for i in self.cursor.description)
                    # print("Header Rows: %s" % [i[0] for i in self.cursor.description])
                    while True:
                        results = self.cursor.fetchmany(row_iterator_size)
                        if not results:
                            break
                        results = [[i.replace('\n', ' ').replace('\r','') if isinstance(i, str) else i for i in r] for r in results]
                        # results = np.array(results).astype(str)
                        # results = [[re.sub(r"\b%s\b" % 'None', '', str(y.encode('utf_8'))[2:len(str(y.encode('utf_8'))) - 1]) for y in x] for x in results]
                        writer.writerows(results)
                        row_count += len(results)
                        print("\rMSSQL query CSV file rows written (%s): %s" % (encoding, row_count), end="")

                print("\nWrite to CSV complete!")
                break
            except IndexError:
                print('All possible encodings exhausted. Throwing error. Check the log file.')
                raise
            except Exception as e:
                print('Use of encoding "%s" to write file failed. Trying another...' % encoding)
                pass

        return None, file_name, encoding


    def closeDriver(self):
        self.conn.close()


if __name__ == '__main__':
    db = query('test')
