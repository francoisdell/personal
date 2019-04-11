__author__ = 'mackea1'
import re, datetime, random, csv, codecs
import numpy as np
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import logging
from collections import OrderedDict

# create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Commit the download directory
import tempfile

tempDir = tempfile.mkdtemp()
filter_set = list()

class query:

    def __init__(self, query_dir: str=None, db_name: str="VMAXAnalytics"):

        self.query_dir = query_dir

        if db_name == 'Sizer':
            self.host = "vmaxdbsp167.isus.emc.com"
            self.database = "vmax"
            self.port = "5432"
            self.user = "vmaxteam"
            self.password = "x7DTc@SQ"

        elif db_name == 'VMAXFAST':
            self.host = "fastdbpc.lss.emc.com"
            self.database = "fast"
            self.port = "5432"
            self.user = "postuser"
            self.password = "x7DTc@SQ"

        elif db_name == 'VMAXAnalytics':
            self.host = "10.246.111.12"
            self.port = "5432"
            self.database = "vmaxanalytics"
            self.user = "automation"
            self.password = "Icoapps1!"
            self.schema_name = "reporting"

        elif db_name == 'RAPID':
            self.host = "bdlgpmcprd01a.isus.emc.com"
            self.port = "6400"
            self.database = "gp_bdl_cust_prod01"
            self.user = "mackea1"
            self.password = "8832bd86"
            self.schema_name = "emcas_iip_pricing"

        else:
            print("Unknown database specified for query! Failed!")
            return

        self.open_pool()

    def open_pool(self):

        self.db = psycopg2.pool.ThreadedConnectionPool(1, 10
                                              , host=self.host
                                              , database=self.database
                                              , user=self.user
                                              , password=self.password
                                              , port=self.port
                                              , connect_timeout=60)
        self.pool_encoding = self.db.getconn().encoding

    def goToQuery(self, query_file):

        if '.sql' in query_file:
            # Commit the download directory
            with open("%s\\%s" % (self.query_dir, query_file), "r") as myfile:
                self.query_text = myfile.read().replace('ï»¿', '').rstrip()
                if self.query_text[-1] == ';':
                    self.query_text = self.query_text[:-1]
        else:
            self.query_text = query_file

    @contextmanager
    def get_cursor(self):
        try:
            con = self.db.getconn()
            con.cursor().execute("SELECT 1")
        except psycopg2.pool.PoolError:
            self.open_pool()
            con = self.db.getconn()
        try:
            yield con.cursor()
            con.commit()
        finally:
            try:
                self.db.putconn(con)
            except psycopg2.pool.PoolError:
                pass

    def runQuery(self, return_table_only: bool=False, query_text: str=None):

        if query_text:
            self.query_text = query_text

        try:
            with self.get_cursor() as cursor:

                # cursor.execute("SHOW server_encoding")
                # server_encoding = cursor.fetchone()[0]
                # print(server_encoding)
                # print(self.db.getconn().encoding)

                if 'INSERT INTO' in self.query_text or 'TABLE' in self.query_text or 'UPDATE' in self.query_text:
                    cursor.execute(self.query_text)
                    printAndLog('Executing INSERT/UPDATE/OTHER query from "runQuery" function.')
                    return True if cursor.description is None else False, None, self.pool_encoding
                elif return_table_only:
                    printAndLog('Executing read-only query from "runQuery" function.')
                    cursor.execute(self.query_text)
                    header = np.array(cursor.description)
                    header = header.transpose()[0]
                    rows = cursor.fetchall()
                    results = np.vstack((header, rows)) if rows else header
                    results = results.astype(str)
                    results = [
                        [re.sub(r"\b%s\b" % 'None', '', str(y.encode('utf-8'))[2:len(str(y.encode('utf-8'))) - 1]) for y
                         in x]
                        for x in results]
                    return results
                else:
                    import tempfile
                    self.tempDir = tempfile.mkdtemp()
                    file_name = self.tempDir + '\\Postgres_Query.csv'
                    with open(file_name, mode='w', encoding='utf_8') as csv_file:
                        query_text = "COPY (%s) TO STDOUT WITH DELIMITER ',' CSV HEADER QUOTE '\"'" % self.query_text
                        # print(query_text)
                        printAndLog('Executing COPY query from "runQuery" function.')
                        cursor.copy_expert(query_text, csv_file)

                    return None, file_name, self.pool_encoding
        except:
            self.closeDriver()
            raise

    def commitQuery(self
                    , data_files: list
                    , table_name: str
                    , query_type: str
                    , run_first: str=None):

        full_table_name = '"%s"."%s"' % (self.schema_name, table_name)

        with self.get_cursor() as cursor:
            cursor.execute("SHOW server_version")
            server_version = cursor.fetchone()[0]
            # print(server_version)

        # Make sure all the files are assigned an encoding
        for f in data_files:
            if not f.encoding:
                f.encoding = get_encoding_type(f.file_addr)

        unique_col_vals = OrderedDict()
        test_file = list(data_files)[0]
        with getCSVIter(test_file.file_addr, test_file.encoding) as row_iter:
            new_cols = [' '.join(i.replace('0xff', '').replace('\\ufeff','').replace('﻿','').split()) for i in next(row_iter)]
            for col_index, col_name in enumerate(new_cols):
                while col_name in unique_col_vals:
                    if col_name[-1].isdigit():
                        col_name = col_name[:-1] + ('%s' % int(col_name[-1:]) + 1)
                    else:
                        col_name += '0'
                unique_col_vals[col_name] = set()

        create_table = True
        old_cols = [c[0] for c in self.get_table_cols(table_name)]
        # Only delete a table or truncate it if truncation is specified and the table exists...
        if self.checkTableExists(full_table_name):
            if query_type in ('truncate', 'append'):  # Only do anything if the table is a truncate table.
                if all(i == j for i, j in zip(old_cols, new_cols)):  # If tables have the same initial col names...
                    create_table = False
                elif query_type == 'append':
                    int_suffix = 1
                    full_table_name = '"%s"."%s%s"' % (self.schema_name, table_name, int_suffix)
                    while self.checkTableExists(full_table_name):
                        int_suffix += 1
                        full_table_name = '"%s"."%s%s"' % (self.schema_name, table_name, int_suffix)
                    create_table = True
                    print("WARNING: CSV and table [%s] fields didn't match on 'append' query.\n"
                          "Creating a new table and leaving the original one untouched."
                          % (table_name))
            elif query_type == 'new': # delete and recreate the table no matter what
                create_table = True
            else:  # If the table is not a truncate table, leave it alone.
                create_table = False

        if create_table:
            final_cols = new_cols
            file_encodings = dict()
            for file_obj in data_files:
                file_path = file_obj.file_addr
                print('Extracting column names from: %s' % file_path)
                file_encodings[file_path] = file_obj.encoding
                print("Encoding for file [%s]: %s" % (file_path.split("\\")[-1], file_encodings[file_path]))

                while True:
                    try:
                        with getCSVIter(file_path, file_encodings[file_path]) as row_iter:
                            next(row_iter)
                            # next(row_iter)
                            print(new_cols)
                            for row in row_iter:
                                # print(row)
                                for col_index, col_name in enumerate(new_cols):
                                    unique_col_vals[col_name].add(row[col_index])
                        break

                    except UnicodeDecodeError as e:
                        if e.encoding == 'utf-8':
                            file_encodings[file_path] = 'latin_1'
                            pass
                        else:
                            print("Exception output below.")
                            print(e.encoding)
                            print(str(e))
                            print(e.reason)
                            raise
                    except Exception as e:
                        print(e)
                        raise
            try:
                print("Determining column data types...")
                fields_list = str()
                for col_name, col_vals in unique_col_vals.items():
                    col_vals.discard('')
                    if len(col_vals) == 0:
                        dtype = 'character varying'
                    elif all(get_data_type(x) == 'int' for x in col_vals):
                        dtype = 'bigint'
                    elif all(get_data_type(x) in ('float', 'int') for x in col_vals):
                        dtype = 'float'
                    elif all(get_data_type(x) == 'date' for x in col_vals):
                        dtype = 'date'
                    else:
                        dtype = 'character varying'

                    fields_list += ('%s"%s" %s' % (', ' if fields_list else '', str(col_name), dtype))

                print("Existing table has different columns. Deleting table [%s]" % full_table_name)
                drop_string = "DROP TABLE IF EXISTS %s" % full_table_name

                # Connect to table (and if "new" is specified as "True", (re)create it first)
                # Create the table if it doesn't already exist
                print("Table doesn't exist. Creating table [%s]" % full_table_name)

                create_string = "CREATE TABLE %s (%s)" % (full_table_name, fields_list)
                print(create_string)

                with self.get_cursor() as cursor:
                    try:
                        cursor.execute(drop_string)
                        cursor.execute(create_string)
                    except psycopg2.DataError as e:
                        self.db.getconn().rollback()
                        printAndLog("Failed to execute drop/create: [%s] table: [%s]" % (self.database, full_table_name), 'info')
                        printAndLog(e, 'exception')
                        raise e
            except:
                self.closeDriver()
                raise

        else:
            final_cols = old_cols

        try:
            print("Inserting data into SQL db: [%s] table: [%s]" % (self.database, full_table_name))

            with self.get_cursor() as cursor:

                if query_type == 'truncate':
                    print('Truncating table [%s]' % full_table_name)
                    if server_version[0] >= '9':
                        drop_string = 'TRUNCATE TABLE ONLY %s RESTART IDENTITY' % full_table_name
                    else:
                        drop_string = 'TRUNCATE TABLE %s' % full_table_name
                    cursor.execute(drop_string)

                if run_first and query_type=='replace':
                    run_first = [run_first] if isinstance(run_first, str) else run_first
                    for q in run_first:
                        cursor.execute(q)

                for file_obj in data_files:
                    file_path = file_obj.file_addr
                    file_encoding = file_obj.encoding

                    with getFile(file_path, file_encoding) as (file, delimiter, quotechar):
                        try:
                            query_text = "COPY %s (%s) FROM stdin WITH DELIMITER '%s' CSV HEADER QUOTE '%s'" \
                                         % (full_table_name, ','.join(['"%s"' % n for n in new_cols]), delimiter, quotechar)
                            print(query_text)
                            # print('Stdin: %s' % file)
                            cursor.copy_expert(query_text, file)
                        except (psycopg2.DataError, psycopg2.ProgrammingError) as e:
                            self.db.getconn().rollback()
                            printAndLog("FAILED to load to db - Incompatible data types: [%s] table: [%s]"
                                        % (self.database, full_table_name), 'info')
                            printAndLog(e, 'error')
                            if query_type == 'append':

                                int_suffix = 1
                                temp_table_name = '%s_temp%s' % (table_name, int_suffix)
                                while self.checkTableExists(temp_table_name):
                                    int_suffix += 1
                                    temp_table_name = '%s_temp%s"' % (table_name, int_suffix)

                                print("WARNING: CSV and table fields didn't match on 'append' query.\n"
                                      "Creating a new table [%s], copying over from the original [%s]"
                                      ", dropping the original, and renaming the new table to the original name."
                                      % (temp_table_name, table_name))

                                fields_list = str()
                                for col in self.get_table_cols(table_name):
                                    fields_list += ('%s"%s" %s' % (', ' if fields_list else ''
                                                       , str(col[0])
                                                       , "character varying" if str(col[0]) in str(e) else str(col[1])))
                                create_string = 'CREATE TABLE "%s"."%s" (%s)' % (self.schema_name, temp_table_name, fields_list)
                                self.runQuery(query_text=create_string)

                                copy_query = 'INSERT INTO "%s"."%s" (%s) SELECT * FROM %s' % \
                                             (self.schema_name
                                              , temp_table_name
                                              , ', '.join(['"' + c + '"' for c in final_cols])
                                              , full_table_name)
                                self.runQuery(query_text=copy_query)
                                self.runQuery(query_text='DROP TABLE IF EXISTS %s' % full_table_name)
                                self.runQuery(query_text='ALTER TABLE IF EXISTS "%s"."%s" RENAME TO "%s"'
                                                       % (self.schema_name, temp_table_name, table_name))
                                return self.commitQuery(data_files, table_name, query_type='append')

                            elif not create_table:
                                if query_type == 'truncate':
                                    printAndLog("Rerunning as delete/reload operation", 'info')
                                    return self.commitQuery(data_files, table_name, query_type='new')
                                else:
                                    printAndLog("Cannot rerun as delete/reload because query type is non-truncate", 'info')
                                    raise e

                        except psycopg2.Error as e:
                            printAndLog("FAILED to load to db: [%s] table: [%s]" % (self.database, full_table_name), 'info')
                            printAndLog(e, 'error')
                            self.db.getconn().rollback()
                            raise e
                        except IndexError as e:
                            printAndLog("FAILED to load to db: [%s] table: [%s]" % (self.database, full_table_name), 'info')
                            printAndLog("Looks like you've got a return code error (e.g. \\n or \\r) screwing up your "
                                        "data.\nConsider introducing a 'replace()' statement somewhere in the data "
                                        "extraction process.")
                            printAndLog(e, 'error')
                            self.db.getconn().rollback()
                            raise e

            with self.get_cursor() as cursor:
                cursor.execute("ANALYZE %s" % full_table_name)

            print("Insert complete!")

        except Exception as e:
            printAndLog("FAILED to load to db and COULD NOT RECOVER. Check the log!!!")
            printAndLog("====   Database: [%s]  ||  Table: [%s]  ====" % (self.database, full_table_name), 'info')
            printAndLog(e, 'error')
            self.closeDriver()
            raise

        return full_table_name

    def get_table_cols(self, table_name):
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM information_schema.columns WHERE table_schema = '%s' AND table_name = '%s'"
                           % (self.schema_name, table_name))
            col_details = [[c[3], c[7]] for c in cursor.fetchall()]
            return col_details

    def check_conn(self):
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
        except psycopg2.pool.PoolError:
            self.open_pool()

    def closeDriver(self):
        if not self.db.closed:
            self.db.closeall()
        # self.conn.close()

    def checkTableExists(self, full_table_name:str):
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT * FROM %s LIMIT 1" % full_table_name)
                return True
        except psycopg2.Error as e:
            return False

def printAndLog(msg, log_type: str=None):
    print(msg)
    if log_type == 'info':
        logger.info(msg)
    elif log_type == 'exception':
        logger.exception(msg)
    if log_type == 'error':
        logger.error(msg)

def get_data_type(x):
    try:
        a = float(x)
        if (not a.is_integer()) or '.' in x:
            return 'float'
        else:
            return 'int'
    except:
        try:
            datetime.datetime.strptime(x, '%Y-%m-%d')
            return 'date'
        except ValueError:
            return 'character varying'

@contextmanager
def getCSVIter(csvFile, encoding: str=None):
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

@contextmanager
def getFile(csvFile, encoding: str=None):
    if not encoding:
        encoding = get_encoding_type(csvFile)
    file = open(csvFile, mode='r', encoding=encoding)
    dialect = csv.Sniffer().sniff(file.readline())
    delimiter = dialect.delimiter
    quotechar = dialect.quotechar
    print("opened file [%s] with encoding [%s], delimiter [%s], & quotechar [%s]"
          % (csvFile.split('\\')[-1], encoding, delimiter, quotechar))
    file.seek(0)
    try:
        yield file, delimiter, quotechar
    finally:
        file.close()

def get_encoding_type(current_file: str):
    printAndLog("Determining file encoding: [%s]" % current_file, 'info')
    from chardet.universaldetector import UniversalDetector
    detector = UniversalDetector()
    detector.reset()
    for line in open(current_file, 'rb'):
        detector.feed(line)
        if detector.done: break
    detector.close()
    print(current_file.split('\\')[-1] + ": " + detector.result['encoding'])
    return detector.result['encoding']

def remove_bad_chars(f):
    return [re.sub(r"\b%s\b" % 'None', '', str(r.encode('utf-8'))[2:len(str(r.encode('utf-8'))) - 1]) for r in f]

class fake_file_obj:
    def __init__(self, file_addr, encoding: str=None):
        self.file_addr = file_addr
        self.encoding = encoding

if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    # db = query(db_name='RAPID')
    db = query(db_name='VMAXAnalytics')
    #
    # dirpath = 'C:\\Users\\Administrator\\Documents\\Automated Reporting\\Analytics Reports\\VMAX Pipeline Report\\Data'
    # data_files = list(fake_file_obj(join(dirpath, f), 'latin_1') for f in listdir(dirpath) if isfile(join(dirpath, f)) and 'All Pipeline' in f)
    # full_table_name = str()
    # table_name = 'pipeline_vmax_all_r3'
    # for f in data_files:
    #     print(f.file_addr)
    #     rptg_date = f.file_addr.split("Date ")[-1].split(")")[0]
    #     if rptg_date >= '2016-08-03':
    #         this_table = db.commitQuery(data_files=[f], table_name=table_name, query_type='append')
    #         if this_table != full_table_name:
    #             full_table_name = this_table
    #             table_name = full_table_name.split(".")[-1].replace('"','')
    #             printAndLog("New Table Name: %s" % (full_table_name))
    #             db.runQuery(query_text="DO $$ BEGIN BEGIN ALTER TABLE %s ADD COLUMN rptg_date date; "
    #                         "EXCEPTION WHEN duplicate_column "
    #                         "THEN RAISE NOTICE 'column rptg_date already exists in %s.'; END; END; $$"
    #                                    % (full_table_name, full_table_name))
    #         db.runQuery(query_text='UPDATE %s SET rptg_date=\'%s\'::date WHERE rptg_date IS NULL' % (full_table_name, rptg_date))

    # dirpath = 'C:\\Users\\Administrator\\Documents\\Automated Reporting\\Data Files\\drive inventory\\'
    # data_files = list(fake_file_obj(join(dirpath, f)) for f in listdir(dirpath) if isfile(join(dirpath, f)))
    # table_name = 'inventory_drives'
    # for f in data_files:
    #     print(f.file_addr)
    #     rptg_date = f.file_addr.split("report_")[-1].split(".")[0]
    #     full_table_name = db.commitQuery(data_files=[f], table_name=table_name, query_type='append')
    #     db.runQuery(query_text="DO $$ BEGIN BEGIN ALTER TABLE %s ADD COLUMN rptg_date date; "
    #                             "EXCEPTION WHEN duplicate_column "
    #                             "THEN RAISE NOTICE 'column rptg_date already exists in %s.'; END; END; $$"
    #                                        % (full_table_name, full_table_name))
    #     db.runQuery(query_text='UPDATE %s SET rptg_date=\'%s\'::date WHERE rptg_date IS NULL' % (full_table_name, rptg_date))

    dirpath = 'C:\\Users\\Administrator\\Documents\\Automated Reporting\\Analytics Reports\\Bulk Loading - Test\\Data\\'
    import os
    os.chdir(dirpath)
    data_files = list(fake_file_obj(join(dirpath, f), encoding='utf_16') for f in listdir(dirpath)
                  if isfile(join(dirpath, f)) and 'VMAX-2016-Q3' in f)
    table_name = 'Orders_Bookings_VMAX'
    db.commitQuery(data_files, table_name, 'append')

    # data_files = set()
    # import os, time
    # days = 0.2
    # for f in os.listdir(dirpath):
    #
    #     now = time.time()
    #     filepath = os.path.join(dirpath, f)
    #     last_modified = os.stat(filepath).st_mtime
    #     if last_modified > now - (days * 86400) and ("Elcid" in f or "SYR" in f):  # 86400 = number of seconds in one day
    #         data_files.add(fake_file_obj(filepath))
    #         print(filepath)


