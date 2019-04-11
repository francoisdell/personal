__author__ = 'mackea1'

# ----------------------------NOTES----------------------------
# Written by Andrew Mackenzie - email: amack87@gmail.com
# Property EMC Corp. Do not re-use without EMC's permission.
# This code was intended for use with 64-bit Windows systems.
# It has only been tested with VMs running Windows 2008 R2.
# Dave Grohl is maybe the coolest motherfucker on earth.
# -------------------------------------------------------------
import Settings
from DataDelivery.SendEmail import sendEmail
import os, importlib, logging, gc, shutil, time, sys, win32com.client, csv
from datetime import date
from datetime import datetime

global base_dir, download_dir, backup_dir, data_file_list, report_name, is_test, logger, attach_setting
global use_whole_directory, start_time, thread_results, failed_files, has_failed, loading_db, vba_dir
global backup_vba_dir

data_file_list = []
attach_list = []
start_time = time.time()
failed_files = []
query_sets_dict = {}
new_query_sets_dict = {}
has_failed = False

def initDriver(reportName,
                isTest: bool=Settings.default_is_test,
                useWholeDirectory: bool=Settings.default_vba_use_whole_directory,
                backupDir: str=False,
                baseDir: str=False,
                downloadDir: str=False,
                default_loading_db: str='VMAXAnalytics'):

    global report_name, base_dir, download_dir, backup_dir, is_test, logger, use_whole_directory, load_to_db, loading_db
    report_name = reportName
    is_test = isTest
    use_whole_directory = useWholeDirectory
    loading_db = default_loading_db

    try:

        logfile_addr = os.path.dirname(os.path.realpath(sys.argv[0])) + '\\logfiles\\logfile_%s.log' % reportName
        with open(logfile_addr, 'w'): pass

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

        printAndLog("\n------------------------NEW EXECUTION------------------------\n", 'info')
        printAndLog("Executing Python File: %s" % (getExecFile()), 'info')

        # create the core variables
        base_dir = baseDir if baseDir else Settings.default_base_dir + '\\%s' % reportName
        backup_dir = backupDir if backupDir else Settings.default_backup_dir + '\\%s' % reportName
        if downloadDir == 'base':
            download_dir = base_dir
        elif downloadDir:
            download_dir = downloadDir
        else:
            download_dir = base_dir + '\\Data'

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        global backup_vba_dir, vba_dir
        backup_vba_dir = backup_dir + Settings.default_vba_dir
        vba_dir = base_dir + Settings.default_vba_dir
        os.makedirs(backup_dir + Settings.default_vba_dir, exist_ok=True)

    except Exception as e:
        fail_and_exit(e)

def terminate_pool(pool):
    pool.terminate()

def runQueries(query_sets: set
               , max_threads: int=Settings.default_max_threads
               , max_wait_time: int=Settings.default_query_wait_time):

    global new_query_sets_dict
    query_threads = []
    new_query_sets_dict.clear()
    for s in query_sets:
        new_query_sets_dict[s.prefix] = s
        for t in s.getQueryThreads():
            query_threads.append(t)

    query_sets.clear() # clears the query sets

    #print(query_threads)
    exception = None
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        global thread_results
        global has_failed
        thread_results = {executor.submit(initializeQuery, args): query_threads for args in query_threads}

        for future in concurrent.futures.as_completed(thread_results):
            thread_result = thread_results[future]
            try:
                return_value = future.result(timeout=max_wait_time)
                try:
                    printAndLog("Thread returned: %s" % return_value, 'info')
                except TypeError as e:
                    pass
            except concurrent.futures.TimeoutError as e:  # thrown if thread takes longer than max_wait_time to finish
                printAndLog("TimeoutError: Thread did not complete within the alloted time limit.", 'error')
                has_failed = True
                exception = e
                pass
            except concurrent.futures.CancelledError as e:  # thrown if an outside source cancels the thread
                printAndLog("CancelledError: Thread was cancelled by the code before completion.", 'error')
                has_failed = True
                exception = e
                pass
            except Exception as e:
                printAndLog('Un-Handled Exception in thread Execution.', 'info')
                has_failed = True
                exception = e
                pass

    printAndLog('All threads completed!!!', 'info')
    if failed_files:
        printAndLog('Failed Files: %s' % failed_files, 'info')
    move_files()

    if has_failed or exception:
        fail_and_exit(exception)
        return False

    return True

def cancelAllThreads():
    global thread_results
    for f in thread_results:
        try:
            if f.running():
                f.cancel()
        except Exception as e:
            pass


def initializeQuery(query_thread):
    printAndLog('Starting Query: %s ||| Prefix: %s' % (query_thread.url, query_thread.prefix),'info')

    try:
        q = Query(query_thread, download_dir)
        return query_thread.prefix
    except Exception as e:
        msg = "Unable to successfully run the query: %s" % (query_thread.prefix if query_thread.prefix else query_thread.url)
        printAndLog(msg, 'info')
        printAndLog(e, 'exception')
        global new_query_sets_dict
        new_query_sets_dict[query_thread.prefix].query_successful = False
        # cancelAllThreads()
        # q.close_query()
        raise e
        #fail_and_exit(e)
    finally:
        q.close_query()

class Query:

    # Initialize the Query object
    def __init__(self, query_thread, download_dir):
        self.url = query_thread.url
        self.filter_set = query_thread.filter_set
        self.query_type = query_thread.query_type
        self.prefix = query_thread.prefix
        self.suffix = query_thread.suffix
        self.download_dir = download_dir
        self.ignore_if_exists = query_thread.ignore_if_exists
        self.column_mapping = query_thread.column_mapping

        # Set the query in motion...
        self.setFiltersAndRun(self.prefix, self.filter_set, self.suffix)

    def close_query(self):
        try:
            self.query.closeDriver()
        except Exception:
            pass


    # If any filters were specified, apply each of them.
    # If there are multiple different filter sets, apply them in each possible combination.
    # After applying each filter, call the "runQuery" function and return the results.
    def setFiltersAndRun(self, prefix: str, filter_set: set = False, suffix: str = False):

        filter_set_list = []
        sql_where_clauses = set()
        new_name = prefix
        # Recursively set filters and run after each filter combination has been set
        if filter_set:
            for filter in filter_set:
                filter_values = filter.values
                filter_operators = filter.operators
                for y in range(len(filter_values)):

                    # Append the filters to the filter set
                    printAndLog("Setting filter: %s: %s" % (filter_values[y], filter_operators[y]))
                    filter_set_list.append([filter.parent, filter.name, filter_values[y], filter_operators[y]])

                    # The next couple lines are to prepare for a 'replace' database insertion operation
                    sql_val_list = ','.join("'%s'" % v for v in filter_values[y])
                    sql_where_clauses.add('"%s" IN (%s)' % (filter.name, sql_val_list))

                    # Modify the final name of the file to describe the filters
                    if isinstance(filter_values[y], (list, tuple)):
                        new_name += '-' + str(filter_values[y][0]) + '-to-' + str(filter_values[y][-1])
                    else:
                        new_name += '-' + str(filter_values[y])

            # Create the delete statement that precedes the insert statement in the 'replace' database insertion operation
            sql_where_clause = ' AND '.join(c for c in sql_where_clauses)
            if not new_query_sets_dict[self.prefix].db_run_first:
                new_query_sets_dict[self.prefix].db_run_first = 'DELETE FROM "reporting"."%s" WHERE (%s)' % (prefix, sql_where_clause)
            else:
                new_query_sets_dict[self.prefix].db_run_first += ' OR (%s)' % sql_where_clause

        new_name += ("-" + suffix if suffix else "")
        new_name = new_name.replace("/", "-")
        new_file_name = (new_name + ('.csv' if not '.csv' in new_name else ''))

        # run the query
        if self.ignore_if_exists:
            possible_file_address = self.download_dir + '\\' + new_file_name
            #print(possible_file_address)
            from os import listdir
            from os.path import isfile, join
            now = time.time()
            matching_files = set()
            for f in listdir(self.download_dir):
                data_file = join(self.download_dir, f)
                if isfile(data_file) \
                        and (new_name in f) \
                        and ('.csv' in f) \
                        and os.stat(data_file).st_mtime < now - self.ignore_if_exists * 3600:
                    matching_files.add(data_file)

            if matching_files:
                printAndLog("File already exists. No need to query. Skipping... (%s)" % possible_file_address, 'info')
                for file_addr in matching_files:
                    print("Adding [%s] to thread [%s]" % (file_addr, self.prefix))
                    new_query_sets_dict[self.prefix].addFile(file_addr, file_addr, None)
                return True

        # Handle case-specific actions depending on the type of query being attempted
        if self.query_type == "OBIEE":
            self.query_lib = importlib.import_module("DataDelivery.QuerySMF")
            self.query = self.query_lib.query()
            if 'http' not in self.url:
                self.url = 'https://entbi.isus.emc.com/analytics/saw.dll?Answers' \
                           '&Path=%2Fusers%2Fma%2Fmackea1%2FAutomated%20Reporting%2F' \
                           + self.url.replace(' ', '%20').replace(',','%2C').replace('\\','%2F').replace('/','%2F')
        elif self.query_type == "OBIEER2":
            self.query_lib = importlib.import_module("DataDelivery.QuerySMFR2")
            self.query = self.query_lib.query()
            if 'http' not in self.url:
                self.url = 'http://corpusapp165:9704/analytics/saw.dll?Answers' \
                           '&_scid=EfLuKk7*KwE&Path=%2Fusers%2Fobiee1%2FVMAX%20Analytics%2F' \
                           + self.url.replace(' ', '%20').replace(',','%2C').replace('\\','%2F').replace('/','%2F')
        elif self.query_type == "BOBJ":
            self.query_lib = importlib.import_module("DataDelivery.QueryBObj")
            self.query = self.query_lib.query()
            if 'http' not in self.url:
                self.url = 'https://sapbip.propel.emc.com/BOE/OpenDocument/opendoc/openDocument.jsp?sIDType=CUID' \
                           '&iDocID=' + self.url
        elif self.query_type in ["Sizer", "VMAXFAST", "VMAXAnalytics"]:
            self.query_lib = importlib.import_module("DataDelivery.QueryPostgres")
            self.query = self.query_lib.query(os.path.expanduser(base_dir + '\\Queries'), self.query_type)
        elif self.query_type in ["SYR", "ElcidStaging", "OPT", "DUDL"]:
            self.query_lib = importlib.import_module("DataDelivery.QueryMSSQL")
            self.query = self.query_lib.query(os.path.expanduser(base_dir + '\\Queries'), self.query_type)
        elif self.query_type == "CSV":
            self.query_lib = importlib.import_module("DataDelivery.QueryCSV")
            self.query = self.query_lib.query()
            if '\\' not in self.url:
                self.url = os.path.dirname(os.path.realpath(sys.argv[0])) + '\\Data Files\\' + self.url
        elif "xls" in self.query_type:
            self.query_lib = importlib.import_module("DataDelivery.QueryXLS")
            if not os.path.exists(self.query_type):
                self.query_type = os.path.dirname(os.path.realpath(sys.argv[0])) + '\\Data Files\\' + self.query_type
            self.query = self.query_lib.query(self.query_type)
        else:
            msg = 'Unable to determine the query type from the query_type field. Check your inputs.'
            printAndLog(msg)
            fail_and_exit(Exception(msg))

        time.sleep(1)
        self.query.goToQuery(self.url)

        time.sleep(1)
        for filter in filter_set_list:
            self.query.setFilter(*filter)

        time.sleep(1)
        self.runQuery(new_file_name)
        return True


    # Run the query
    def runQuery(self, new_name):

        new_file = download_dir + '\\' + new_name + ('.csv' if not '.csv' in new_name else '')

        result_data, original_file, encoding = self.query.runQuery()

        try:
            # If results were found, change the filename
            if result_data is True:
                printAndLog("Non-data-generating query successful!" % new_name, 'exception')
                return
            if result_data and not original_file:
                import tempfile
                #original_file_name = tempfile.mkdtemp() + '\\' + new_name + ('.csv' if not '.csv' in new_name else '')
                original_file, encoding = self.writeCSV(result_data, new_file, encoding)  # USE THIS TO CREATE A NEW CSV FILE
            if original_file:
                global new_query_sets_dict
                printAndLog("Valid data file recieved! Modifying...",'info')
                if isinstance(original_file, (list, tuple)):
                    for index, orig_file in enumerate(original_file):
                        new_query_sets_dict[self.prefix].addFile(orig_file
                                                                 , new_file.replace('.csv', '(%s).csv' % index)
                                                                 , None if not encoding else encoding[index])
                else:
                    new_query_sets_dict[self.prefix].addFile(original_file
                                                             , new_file.replace('.csv', '(0).csv')
                                                             , encoding)
            else:
                globals()['failed_files'].append([new_file])
                printAndLog("Query for %s did not return any results. Log and continue." % new_file, 'exception')
                return
        except Exception as e:
            raise e

    # Write the result_data (which should be a list or other subscriptable, iterable item) to a .csv file.
    def writeCSV(self, result_data, new_file_addr, encoding: str=None):

        # the following "if" statement will reorder the columns in the result data using the provided column_mapping
        # it will also rename them to a set of new names specified in the column_mapping
        if self.column_mapping:
            orig_headers = result_data[0]
            if len(result_data[0]) == len(self.column_mapping):
                new_order = []
                for k in list(self.column_mapping.keys()):
                    new_order.append(result_data[0].index(k))
                #for h in result_data[0]:
                #    new_order.append(list(self.column_mapping.keys()).index(h))
                #print(new_order)
                result_data = [[row[ci] for ci in new_order] for row in result_data] # rearrange the column orders
                result_data[0] = list(self.column_mapping.values()) # rename the columns to the new names
                #print(result_data[:3])
            else:
                print("orig_headers: %s rows || column_mapping: %s rows" % (len(result_data[0]),len(self.column_mapping)))
                msg = 'Ordered column list does not match the result data. Cannot sort the columns. Ending.'
                raise Exception(msg)
                #fail_and_exit(Exception(msg))

        # Write the result_data to file. This will cycle through all provided encodings (below) from right to left,
        # until it finds an encoding that succeeds. It will only fail if all encodings fail.
        # If one encoding fails but others remain to be tried, it will try those until it succeeds or until none remain.
        # If none remain, this will throw an exception and the program will fail.
        from collections import deque
        possible_encodings = deque(['ascii', 'utf_32', 'utf_8', 'utf_16_le', 'utf_16', 'cp1252', 'latin_1'])
        printAndLog("Encoding and writing to: %s" % new_file_addr)
        #printAndLog("Data Sample: %s" % result_data[:2])
        #if os.path.exists(new_file_addr):
        #    os.remove(new_file_addr)
        while True:
            try:
                if not encoding:
                    printAndLog("Trying Encoding: %s" % encoding)
                    encoding = possible_encodings.pop()
                with open(new_file_addr, "w", newline="\n", encoding=encoding) as f:
                    if encoding.upper().replace('-', '').replace('_', '') in ('UTF8', 'LATIN1', 'CP1252'):
                        dialect = csv.excel
                    else:
                        dialect = csv.excel_tab
                    writer = csv.writer(f, dialect=dialect)
                    writer.writerows(result_data)
                printAndLog("Encoding Selected: %s" % encoding)
                break
            except IndexError as e:
                printAndLog('All possible encodings exhausted. Throwing error. Check the log file.')
                raise e
                #fail_and_exit(e)
            except Exception as e:
                printAndLog('Use of encoding "%s" to write file failed. Trying another...' % encoding)
                printAndLog(e)
                encoding = None
                pass

        return new_file_addr, encoding

# Accepts a string "msg" and prints it and/or appends it to the log file.
def printAndLog(msg, log_type: str=False):
    print(msg)
    if log_type == 'info':
        logger.info(msg)
    elif log_type == 'exception':
        logger.exception(msg)
    if log_type == 'error':
        logger.error(msg)


# Modify a file by renaming it and copying it to a new location,
# in this case, the //data folder in the base directory for this report
def modifyFile(orig_file, new_file, encoding):

    if orig_file == new_file:
        return orig_file
    printAndLog("Renaming: [%s] to [%s]" % (orig_file, new_file), 'info')
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
                printAndLog("Unable to delete existing file")
                raise
    try:
        with open(orig_file, mode='r', encoding=encoding) as file:
            dialect = csv.Sniffer().sniff(file.readline())
            dialect.doublequote = True
            data_iter = csv.reader(file, dialect=dialect)
            file.seek(0)
            if encoding.upper().replace('-','').replace('_','') in ('UTF8', 'LATIN1', 'CP1252'):
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
        printAndLog("shutil.move failed. WTF?!?!?")
        printAndLog(e)
        fail_and_exit(e)

    return new_file

# Cleans out all the temporary directories which were created in the temp folder during the file download process.
def move_files():

    global new_query_sets_dict
    #print("New Files:\n%s" % list(new_query_sets_dict.values()))
    for prefix, set in new_query_sets_dict.items():
        query_sets_dict[prefix] = set
        if set.query_successful:
            for orig_file, csv_file in set.result_files.items():
                if not csv_file.encoding:
                    csv_file.encoding = get_encoding_type(orig_file)
                data_file_list.extend([modifyFile(orig_file, csv_file.file_addr, csv_file.encoding)])
        for orig_file in set.result_files.keys():
            orig_dir = os.path.dirname(orig_file)
            if not orig_dir == download_dir or not set.query_successful:
                try:
                    shutil.rmtree(orig_dir)
                    printAndLog("Deleted the temp folder: %s" % orig_dir, 'info')
                except Exception as e:
                    printAndLog("Unable to delete the temp folder: %s" % orig_dir, 'info')
                    # printAndLog(e, 'info')
                    pass
            else:
                printAndLog("Didn't need to delete the temp folder. Was already in download_dir: %s" % orig_dir, 'info')

# Deletes any old queries in the folder
def deleteQueries(values=False):

    if values:
        if not isinstance(values, (list, tuple)):
            values = [values]
        for f in data_file_list:
            values.append(f)
    else:
        values = data_file_list

    for f in os.listdir(download_dir):
        printAndLog('Checking validity of existing file: %s ...' % f)
        for v in values:
            #printAndLog('   ...against data_file_list file: %s' % v.split('\\')[-1])
            if v.split('\\')[-1] in f:
                break
        else:
            printAndLog('Deleting File: %s' % f)
            os.remove(download_dir + '\\' + f)

# Run Excel and send an email out to all users of the report
def closeDriver(users
                , attachReport=False
                , runExcel=True
                , reports_list=list()):

    # The following section will insert your report data into the DB, if desired.
    # It will also log the results of the report runs in the DB, for review and analysis.
    import DataDelivery.QueryPostgres as vmaxanalytics
    vmaxdb = vmaxanalytics.query()
    query_status_table = list()
    query_status_table.append(['timestamp', 'report_name', 'prefix', 'suffix', 'attach', 'query_type', 'ignore_if_exists'
                                  , 'query_successful', 'load_to_db', 'load_successful'])

    datetime_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    global query_sets_dict
    for q in query_sets_dict.values():
        if q.load_to_db and q.query_successful:
            try:
                global loading_db
                load_db = vmaxanalytics.query(db_name=q.loading_db if q.loading_db else loading_db)

                if isinstance(q.db_run_first, dict):
                    queries = list()
                    for k, v in q.db_run_first.items():
                        if k == 'DELETE':
                            queries.append('DELETE FROM "%s"."%s" WHERE "%s" IN (\'%s\')'
                                        % (load_db.schema_name, q.prefix, v[0], "','".join(v[1]) if isinstance(v[1], (list, tuple)) else v[1]))
                        q.db_run_first = queries

                commit_tbl = load_db.commitQuery(data_files=q.result_files.values()
                            , table_name=q.prefix
                            , query_type=q.load_to_db
                            , run_first=q.db_run_first)
                for k, v in q.append_values.items():
                    load_db.runQuery(query_text="DO $$ BEGIN BEGIN ALTER TABLE %s ADD COLUMN %s %s; "
                                    "EXCEPTION WHEN duplicate_column "
                                    "THEN RAISE NOTICE 'column %s already exists in %s.'; END; END; $$"
                                    % (commit_tbl, k, v[0], k, commit_tbl))
                    load_db.runQuery(query_text='UPDATE %s SET "%s"=\'%s\' WHERE "%s" IS NULL'
                                    % (commit_tbl, k, v[1], k))
            except Exception as e:
                q.load_successful = False
                printAndLog("Failed to load data to a table to the DB!! [%s]" % q.prefix)
                printAndLog(e)
                raise e
        elif not q.query_successful:
            q.load_successful = None

        query_status_table.append([datetime_now, report_name, q.prefix, q.suffix, q.attach, q.query_type, q.ignore_if_exists
                                , q.query_successful, q.load_to_db, q.load_successful])

    # Write the query_status_table object to disk and then load it to the database
    global download_dir
    query_status_file_addr = os.path.dirname(os.path.realpath(sys.argv[0])) + '\\Data Files\\query_status_table.csv'
    query_status_file_encoding = 'utf_16'
    with open(query_status_file_addr, "w", newline="\n", encoding=query_status_file_encoding) as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerows(query_status_table)
    query_status_file = CSV_File(query_status_file_addr, query_status_file_encoding)

    vmaxdb.commitQuery(data_files=[query_status_file]
                   , table_name='report_history'
                   , query_type='append')

    # Close the database connection
    vmaxdb.closeDriver()

    attached_files = []
    global has_failed
    if not has_failed:
        try:
            if runExcel:
                global base_dir, report_name, is_test, vba_dir
                # Run the datafile VBA
                if not reports_list:
                    reports_list.append(report_name)
                xl_results = []
                for r in reports_list:
                    vbafile = vba_dir + "\\%s (VBA).xlsm" % r
                    xlfile = base_dir + '\\%s.xlsx' % r
                    print("Running: %s" % vbafile)
                    print("Target: %s" % xlfile)

                    xl_results.append(runDashboard(vba_file=vbafile, result_file=xlfile))

                    if xl_results[-1]:
                        if attachReport:
                            attached_files.append(xlfile)
                        if isinstance(xl_results[-1],(list, tuple)):
                            [attached_files.append(x) for x in xl_results[-1]]
                        printAndLog('Call to %s completed.' % r, 'info')
                    else:
                        printAndLog('Call to %s failed.' % r, 'info')

                failure_list = []
                for r in range(len(xl_results)):
                    if not xl_results[r]:
                        failure_list.append(reports_list[r])
                if failure_list:
                    msg = 'Check the VBA log file(s). The following VBA calls to Excel returned FALSE:\n%s' % failure_list
                    raise Exception(msg)
                    #fail_and_exit(msg)
            purgeDir(backup_dir, Settings.default_backup_period_days)  # delete any files older than 14 days in the backup folder

            setSecurity(os.path.expanduser(base_dir), users)

            if is_test or not users:
                addresses=['Andrew Mackenzie@emc.com']
            else:
                addresses = [x.email for x in users]

            sendMail(success=True
                 , addr=addresses
                 , attach=attached_files)

        except Exception as e:
            #printAndLog(e, 'exception')
            raise e
            #fail_and_exit(e)

    global start_time
    printAndLog("--- Duration: %s minutes ---" % round((time.time() - start_time)/60,2), 'info')
    printAndLog("\n------------------------END EXECUTION------------------------\n", 'info')

def get_encoding_type(current_file):
    from chardet.universaldetector import UniversalDetector
    detector = UniversalDetector()
    detector.reset()
    for line in open(current_file, 'rb'):
        detector.feed(line)
        if detector.done: break
    detector.close()
    print(current_file.split('\\')[-1] + ": " + detector.result['encoding'])
    return detector.result['encoding']

# accepts a list of users (see the User class) and uses each user's email, ntid, and permissions level to share the
# report folder with them over the network.
def setSecurity(folder, users):

    del_cmd = "net share \"%s\" /DELETE /Y" % report_name
    share_cmd = "net share \"%s\"=\"%s\"" % (report_name, folder)

    for u in users:

        if u.level == 'readonly':
            share_cmd += " /GRANT:\"%s\",%s" % ("corp\\" + u.ntid, "READ")
        elif u.level == 'change':
            share_cmd += " /GRANT:\"%s\",%s" % ("corp\\" + u.ntid, "CHANGE")
        elif u.level == 'admin':
            share_cmd += " /GRANT:\"%s\",%s" % ("corp\\" + u.ntid, "FULL")
        else:
            msg = "Unknown permissions level passed: %s. Can't set permissions. Aborting." % u.level
            fail_and_exit(Exception(msg))
    try:
        import subprocess
        print("Delete Command: %s" % del_cmd)
        print("Share Command: %s" % share_cmd)
        subprocess.call(del_cmd)
        time.sleep(3)
        subprocess.call(share_cmd)
    except Exception:
        msg = "Unable to set sharing for user %s. Please ensure that you've provided the correct NTID." % u.ntid
        fail_and_exit(Exception(msg))
        raise

def runDashboard(vba_file, result_file):
    printAndLog('Running Excel file...','info')

    # The below folder creation is required on all windows systems. In order to run Excel from python via the
    # Task Scheduler, this folder must be created and the permissions of the folder must give full access to the
    # user account which Task Scheduler uses to call the python application.

    import platform
    if "Windows" in platform.architecture()[1]:
        if '64bit' in platform.architecture()[0]:
            os.makedirs("C:\\Windows\\SysWOW64\\config\\systemprofile\\Desktop", exist_ok=True)
        elif '32bit' in platform.architecture()[0]:
            os.makedirs("C:\\Windows\\SysWOW64\\config\\systemprofile\\Desktop", exist_ok=True)
    else:
        msg = "Incorrect Operating System. Substring 'Windows' not found in system architecture value returned " \
              "by 'platform.archirecture()[1]' call. Check your 'platform.archirecture()' tuple."

        fail_and_exit(OSError(msg))

    try:
        # Initialize Excel before opening the workbook you want to run
        xl = win32com.client.DispatchEx("Excel.Application")
        xl.Interactive = False
        xl.DisplayAlerts = False

    except Exception as e:
        logger.exception(e)
        print('Failed to open Excel! Check the log file...')
        xl.Quit()
        gc.collect()
        fail_and_exit(e)

    try:
        # Print all the collected data files
        printAndLog("--------Data Files--------",'info')
        global data_file_list
        #print(data_file_list)
        if not isinstance(data_file_list, (list, tuple)):
            data_file_list = [data_file_list]
        if data_file_list:
            for f in data_file_list:
                printAndLog(f, 'info')
        else:
            printAndLog("No Data Files", 'info')
        printAndLog("--------------------------", 'info')

        # Open the workbook(s) and run the VBA code to ingest the list of data files
        if os.path.isfile(vba_file):
            import stat
            os.chmod(vba_file, stat.S_IWRITE)
            wb = xl.Workbooks.Open(Filename=vba_file, ReadOnly=1)
            vbaCall = xl.Application.Run("StartImport", download_dir if use_whole_directory else data_file_list, result_file)
        else:
            msg = 'File %s not found. Something is wrong with the file reference or your permissions...'
            fail_and_exit(FileNotFoundError(msg))

        # If the VBA call returns True or a file, consider it as having succeeded. Otherwise, treat it as a failure.
        if isinstance(vbaCall, bool) and vbaCall:
            printAndLog('VBA StartImport call completed successfully!', 'info')
        elif isinstance(vbaCall, str):
            printAndLog('VBA StartImport call completed successfully and returned a file!', 'info')
        elif isinstance(vbaCall, (list, tuple)):
            printAndLog('VBA StartImport call completed successfully and returned a set of files!', 'info')
        else:
            # If the VBA call failed, close the workbook and quit excel, returning false
            if isinstance(vbaCall, bool):
                printAndLog('VBA StartImport call returned False! Run failed! Closing and aborting!', 'info')
            else:
                printAndLog('VBA StartImport call returned an unknown object! Closing and aborting!', 'error')
            wb.Close(False)
            xl.Quit()
            gc.collect()
            return False

        # Close the workbook and quit excel
        wb.Close(False)
        xl.Quit()
        gc.collect()

    except Exception as e:
        logger.exception(e)
        print('Dashboard run failed! Check the log file...')
        if wb:
            wb.Close(False)
        if xl:
            xl.Quit()
        gc.collect()
        fail_and_exit(e)

    try:
        # Copy the Excel report and VBA files to the backup directory, for safe keeping.
        printAndLog('Copying File...', 'info')

        today_backup_dir = backup_dir + '\\%s' % str(date.today())
        today_backup_dir_data = today_backup_dir + Settings.default_data_dir
        today_backup_dir_vba = today_backup_dir + Settings.default_vba_dir

        try:
            shutil.rmtree(today_backup_dir)
        except FileNotFoundError as e:
            pass

        os.makedirs(today_backup_dir, exist_ok=True)
        os.makedirs(today_backup_dir_vba, exist_ok=True)
        os.makedirs(today_backup_dir_data, exist_ok=True)

        shutil.copy2(result_file, today_backup_dir)
        shutil.copy2(vba_file, today_backup_dir_vba)
        for f in data_file_list:
            shutil.copy2(f, today_backup_dir_data)

    except Exception as e:
        fail_and_exit(e)

    if isinstance(vbaCall, str):
        return [vbaCall]
    elif isinstance(vbaCall, (list, tuple)):
        return vbaCall
    else:
        return True

def fail_and_exit(e: Exception):
    global has_failed
    if not has_failed:
        has_failed = True
        cancelAllThreads()
        printAndLog(e, 'exception')
        sendMail(success=False)
        raise e
        #os._exit(1)  # causes issues with open files - prevents "finally" statements from running

def purgeDir(path, days):
    printAndLog("Scanning %s for files older than %s days." % (path, days))
    for f in os.listdir(path):

        now = time.time()
        filepath = os.path.join(path, f)
        last_modified = os.stat(filepath).st_mtime
        if last_modified < now - (days * 86400): # 86400 = number of seconds in one day
            printAndLog("Removing backup older than %s days: %s" % (f, days))
            shutil.rmtree(filepath, ignore_errors=True)
            print('Deleted: %s (Last modified on: %s)' % (f, date.fromtimestamp(last_modified)))


# prodName -> lookup for column name in the db
# this won't change anything right now, but if you decide to use it will no longer have to supply addresses to sendMail

from leo_fixes.query_server import ServerQuery

def sendMail(success=True, addr=None, attach=False):

    # Any time you send an email, you'll want to remove the temporary directories created for Firefox
    # downloads first for cleanup. Ideal to put here because I basically send an email WHENEVER there
    # is a failure or success of the whole application. And I'd only want to clean up the temporary
    # directories if I was closing the entire application...

    # Set the email subject and text...
    if success:
        if not addr:
            sq = ServerQuery()
            addr = sq.subscription_query(report_name)
            sq.quit()

        base_addr = "\\\\vmaxpaa-prod\\" + report_name

        msg = "<<< Note: This is an auto-generated message >>> <br>" \
            "%s%s update SUCCESSFUL! <br>" \
            "Please see the share at: <a href=\"%s\">%s\\</a> <br>" \
            "Thanks!" \
            % (report_name, ' report' if 'REPORT' not in report_name.upper() else '', base_addr, base_addr)

        subject = '%s Refresh Successful' % report_name

        printAndLog("Report Run SUCCESSFUL!!!",'info')

        # Add the attachments...
        global attach_list
        if attach != False:
            if type(attach) not in (tuple, list):
                attach_list.append(attach)
            else:
                attach_list += attach

    else:
        msg = "<<< Note: This is an auto-generated message >>> <br>" \
            "%s refresh FAILED. Please see the logger for details. <br>" \
            "Thanks!" % report_name

        subject = '%s Refresh Failed' % report_name
        addr=['andrew.mackenzie@emc.com']
        printAndLog("Report Run FAILED!!!",'error')
        attach_list = []

    # Send the email!!!
    sendEmail(subject, msg, addr, attachments=attach_list)

if __name__ == '__main__':
    sq = ServerQuery()
    andy = sq.subscription_query("admin", True)
    sq.email_query(andy, True)
    sq.quit()

def getExecFile():
    # Obtains the name of the python file which was originally executed
    try:
        sFile = os.path.abspath(sys.modules['__main__'].__file__)
    except:
        sFile = sys.executable
    sFile = os.path.splitext(sFile.split('\\')[-1])[0]

    return sFile

# Thread_OBIEE class, which is basically just a holder for the properties of a thread for one or more OBIEE queries
class Thread_OBIEE:
    def __init__(self,
                url,
                prefix,
                filter_set={},
                suffix: str = False,
                attach: bool = False,
                ignore_if_exists: bool = Settings.default_ignore_if_exists):

        self.url = url
        self.prefix = prefix
        self.filter_set = filter_set
        self.suffix = suffix
        self.attach=attach

        self.ignore_if_exists = ignore_if_exists
        self.new_file_list = []

        self.query_type = "OBIEE"

    def return_args(self):
        return self.url, self.prefix, self.filter_set, self.suffix, self.attach, self.query_type, self.ignore_if_exists

    def add_filter(self, filter):
        self.filter_set[filter.name] = filter

# Filter_OBIEE class, which is basically just a holder for the properties of a filter for an OBIEE query
class Filter_OBIEE:
    def __init__(self,
               parent: str,
               name: str,
               values: [],
               operators: []):
        self.parent = parent
        self.name = name
        self.values = values
        self.operators = operators

class Query_Thread:
    def __init__(self,
                 url: str,
                 prefix: str,
                 suffix: str = False,
                 attach: bool = False,
                 query_type: str = "OBIEE",
                 ignore_if_exists: bool = Settings.default_ignore_if_exists,
                 column_mapping: [] = False,
                 filter_set: [] = False):

        self.url = url
        self.prefix = prefix
        self.suffix = suffix
        self.attach = attach
        self.query_type = query_type
        self.ignore_if_exists = ignore_if_exists
        self.column_mapping = column_mapping
        self.filter_set = filter_set

class CSV_File:
    def __init__(self
                 ,file_addr: str
                 ,encoding: str=None):
        self.file_addr = file_addr
        self.encoding = encoding

class Query_Set:
    def __init__(self,
            url: str,
            prefix: str,
            suffix: str = False,
            attach: bool = False,
            query_type: str = "OBIEE",
            ignore_if_exists: bool = False,
            column_mapping: [] = False,
            load_to_db: str = None,
            loading_db: str = None,
            encoding: str = None,
            append_values: dict = {},
            db_run_first = None):

        self.url = url
        self.prefix = prefix
        self.suffix = suffix
        self.attach = attach
        self.query_type = query_type
        self.ignore_if_exists = ignore_if_exists
        self.column_mapping = column_mapping
        self.filter_sets = list()
        self.result_data = list()
        self.result_files = dict()
        self.load_to_db = load_to_db
        self.loading_db = loading_db
        self.load_successful = True if load_to_db else None
        self.query_successful = True
        self.encoding = encoding
        self.db_run_first = db_run_first
        self.append_values = append_values

    def addFilterSet(self, filterset: []):
        self.filter_sets.append(filterset)

    def addData(self, table: list):
        # removes the header before appending if the result_data has already been instantiated
        self.result_data.extend(table[1 if self.result_data else 0:])

    def addFile(self, orig_file: str, new_file: str, encoding: str):
        # removes the header before appending if the result_data has already been instantiated
        self.result_files[orig_file] = CSV_File(new_file, encoding if encoding else self.encoding)

    def getQueryThreads(self):
        if self.filter_sets:
            return [Query_Thread(self.url
                    ,self.prefix
                    ,self.suffix
                    ,self.attach
                    ,self.query_type
                    ,self.ignore_if_exists
                    ,self.column_mapping
                    ,filter_set) for filter_set in self.filter_sets]
        else:
            return [Query_Thread(self.url
                    ,self.prefix
                    ,self.suffix
                    ,self.attach
                    ,self.query_type
                    ,self.ignore_if_exists
                    ,self.column_mapping
                    ,False)]