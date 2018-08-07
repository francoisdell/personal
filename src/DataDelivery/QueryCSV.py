import os, sys, tempfile
# set global variables
default_download_dir = os.path.expanduser('~/Downloads')
os.makedirs(default_download_dir, exist_ok=True)

class query:

    def __init__(self):

        # Commit the download directory
        self.tempDir = tempfile.mkdtemp()

    def goToQuery(self, file_addr: str):
        self.file_addr = file_addr

    def run_query(self):
        import shutil
        tmp_file_addr = shutil.copy2(self.file_addr, self.tempDir)
        print('Copied CSV file: [%s] to [%s]' % (self.file_addr, tmp_file_addr))
        return None, tmp_file_addr, None

    def close_driver(self):
        pass
