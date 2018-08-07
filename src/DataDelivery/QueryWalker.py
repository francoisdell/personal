__author__ = 'mackea1'
import logging
import os
import tempfile
import time
from random import randint

from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from DataDelivery import SSO_SignIn
from selenium.common.exceptions import TimeoutException
from selenium import webdriver

# set global variables
default_downloaddir = os.path.expanduser('~/Downloads')
os.makedirs(default_downloaddir, exist_ok=True)

class query:

    def __init__(self):

        # create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        # Commit the download directory

        self.tempDir = tempfile.mkdtemp()
        self.filter_set = list()
        self.init_driver()

    def init_driver(self, retries=1):
        # Set the firefox profile. This will instruct FireFox to download files automatically, rather than prompting the user
        try:
            fp = webdriver.FirefoxProfile()
            fp.set_preference("browser.download.folderList", 2)
            fp.set_preference("browser.download.panel.shown", False)
            fp.set_preference("browser.download.dir", self.tempDir)
            fp.set_preference("browser.download.downloadDir", self.tempDir)
            fp.set_preference("browser.download.defaultFolder", self.tempDir)
            fp.set_preference("browser.download.manager.showWhenStarting", False)
            fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/csv")
            fp.set_preference("browser.helperApps.alwaysAsk.force", False)
            fp.set_preference("browser.newtab.url", "about:blank")
            fp.set_preference("browser.tabs.animate", False)
            fp.set_preference("browser.cache.use_new_backend", 1)
            fp.set_preference("network.http.pipelining", True)
            fp.set_preference("network.http.pipelining.aggressive", True)
            fp.set_preference("network.http.pipelining.maxrequests", 8)
            fp.set_preference("network.http.pipelining.ssl", True)
            fp.set_preference("toolkit.startup.max_resumed_crashes", "-1")

            # Open the Firefox window and set the profile
            gecko_file = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('\\')) + '/geckodriver.exe'
            print(gecko_file)
            self.driver = webdriver.Firefox(firefox_profile=fp, executable_path=gecko_file)

        except Exception as e:
            if retries > 0:
                self.print_and_log("Failed to load the firefox profile! "
                                 "Might've been trying to load too many at once. Trying again...")
                self.init_driver(retries=retries-1)
            else:
                self.print_and_log("Failed to load the firefox profile AGAIN! WTF!?!?")
                raise e

    # Check download status
    def checkdownload(self, dir: str, retries: int=2):
        os.chdir(dir)
        try:
            files = filter((lambda x: os.path.isfile and os.path.getsize(x) > 0), os.listdir(dir))
            files = [os.path.join(dir, f) for f in files] # add path to each file
            files.sort(key=lambda x: os.path.getmtime(x))
        except Exception as e:
            if retries > 0:
                time.sleep(0.5)
                return self.checkdownload(dir, retries=retries - 1)
            else:
                self.logger.error(e)
                raise e
        if not files:
            return None
        else:
            newest_file = files[-1]
        os.chdir(dir)
        return newest_file

    def goToQuery(self, webAddress: str, retries: int=3):

        s = SSO_SignIn.SSO(self.driver)
        while retries > 0:
            try:
                s.proactiveLogin()
                time.sleep(1)
                self.web_address = webAddress
                self.print_and_log('Login Confirmed', 'info')
                return
            except TimeoutException:
                self.print_and_log('Login Failed. Trying again...', 'info')
                retries -= 1
                pass

        self.close_driver()
        raise Exception("Couldn't log into OBIEE through SSO. Failed %s times. Please fix..." % retries)

    def run_query(self, retries=2):

        self.print_and_log("NEW QUERY EXECUTION ||  %s" % (self.web_address), 'info')
        previousnew = self.checkdownload(self.tempDir)
        # Check if file has begun downloading
        self.driver.get(self.web_address)

        email_field = self.findElement("//input[@id='email']", timeoutLimit=1)
        email_field.send_keys('andrew.mackenzie@emc.com')
        btn = self.findElement("//button[@id='continue']", timeoutLimit=1)
        btn.click()

        self.print_and_log("Waiting for new file in: [%s]" % self.tempDir, 'info')
        new = self.checkdownload(self.tempDir)
        while previousnew == new:
            time.sleep(1)
            print("... waiting")
            new = self.checkdownload(self.tempDir)
            continue

        # New file found, wait until it doesn't have .part extension
        self.print_and_log("Waiting for download to finish in: [%s]" % self.tempDir, 'info')
        new = self.checkdownload(self.tempDir)
        while new is None or os.path.splitext(new)[1] == ".part":
            time.sleep(1)
            print("... downloading")
            try:
                new = self.checkdownload(self.tempDir)
            except Exception as e:
                self.print_and_log("This should be rare. The download failed.")
                self.print_and_log(e)
                if retries > 0:
                    retries -= 1
                    pass
                else:
                    raise e
            continue

        print("Downloaded")
        return None, new, None


    def findElement(self, xpathText, refresh=False, all=False, resetFirst=False, timeoutLimit=3, waitTime=5):
        timeouts = 0
        while True:
            try:
                if resetFirst:
                    self.resetDriver()
                #print("Searching for Element: %s" % xpathText)
                self.logger.info("Searching for Element: %s" % xpathText)
                while True:
                    try:
                        el = WebDriverWait(self.driver, waitTime)\
                                .until(EC.presence_of_all_elements_located((By.XPATH, xpathText)))

                        #print("Element found!")
                        #self.logger.info("Element found!")
                        if all:
                            return el
                        else:
                            return el[0]

                    except TimeoutException as e:
                        if self.driver.execute_script('return document.readyState;') == 'interactive':
                            continue
                        else:
                            raise

            except TimeoutException as e:
                timeouts += 1
                # self.print_and_log("TimeoutException while searching for element %s | Total timeouts: %s/%s" %
                #         (xpathText, timeouts, timeoutLimit), 'info')
                if timeouts < timeoutLimit:
                    if refresh:
                        self.driver.refresh()
                    pass
                else:
                    self.print_and_log("TimeoutException while searching for element %s | Timeout limit exceeded: %s/%s" %
                                     (xpathText, timeouts, timeoutLimit), 'exception')
                    raise
            except UnexpectedAlertPresentException as e:
                self.print_and_log("Couldn't get the XML data. Jumped the gun there, Selenium...")
                pass
            except Exception as e:
                self.print_and_log("CRITICAL failure in the findElement function", 'info')
                self.print_and_log(e, 'error')
                self.logger.exception(e)
                raise

    def openCSV(self, csv_file):
        import csv
        encoding = self.get_encoding_type(csv_file)
        with open(csv_file, mode='r', encoding=encoding) as file:
            dialect = csv.Sniffer().sniff(file.readline())
            dialect.doublequote = True
            data_iter = csv.reader(file, dialect=dialect)
            file.seek(0)
            return list(r for r in data_iter)

    def close_driver(self):
        self.print_and_log("Closing Driver")
        if self.driver:
            self.driver.close()
            self.driver.quit()
            self.driver = None

    def print_and_log(self, msg, log_type=False):
        print(msg)
        if log_type == 'info':
            self.logger.info(msg)
        elif log_type == 'exception':
            self.logger.exception(msg)
        if log_type == 'error':
            self.logger.error(msg)

    def get_encoding_type(self, current_file):
        from chardet.universaldetector import UniversalDetector
        detector = UniversalDetector()
        detector.reset()
        for line in open(current_file, 'rb'):
            detector.feed(line)
            if detector.done: break
        detector.close()
        print(current_file.split('\\')[-1] + ": " + detector.result['encoding'])
        return detector.result['encoding']