import os


root_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).replace('\\', '/').split('/')[:-1])
root_query_dir = root_dir + '/Queries'
root_data_dir = root_dir + '/Data Files'
default_base_dir = root_dir + '/Analytics Reports'
default_backup_dir = root_dir + '/Backup Files'
default_data_dir = '/Data'  # Will be appended to the end of the user-defined base directory for the project
default_vba_dir = '/VBA'  # Will be appended to the end of the user-defined base directory for the project
default_query_dir = '/Queries'  # Will be appended to the end of the user-defined base directory for the project
default_model_dir = 'Models'  # Will be appended to the end of the user-defined base directory for the project
default_query_wait_time = 5400  # 3600 = 1 hour
default_max_threads = 5  # Max simultaneous query threads that can be run at once
default_backup_period_days = 14  # Number of days to retain backup files before deleting
default_loading_db = 'VMAXAnalytics'
default_vba_use_whole_directory = False
default_is_test = False
default_ignore_if_exists = True
default_ignore_if_queried_within_hrs = 11