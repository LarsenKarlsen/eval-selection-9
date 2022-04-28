import pandas as pd
from os import mkdir, path, getcwd
from pandas_profiling import ProfileReport


def create_report_folder():
    '''
    Create report folder in project root directory
    '''
    wd = getcwd() # get current working dir
    report_dir_path = path.join('..',wd,'reports')
    
    if path.exists(report_dir_path):
        return True
    else:
        mkdir(report_dir_path)


def raw_train_profiling_report(dataset_path:str='data//raw//train.csv')->ProfileReport:
    '''
    Create pandas-profiling report from raw train data in project folder
    reports/raw_train_profiling_report.html
    '''
    create_report_folder()
    raw_data_path = path.join('..',getcwd(), dataset_path)
    report_path = path.join('..',getcwd(),'reports','raw_train_profile_report.html')
    
    # load train data into pd.DataFrame
    df = pd.read_csv(raw_data_path, index_col='Id')
    profiling = ProfileReport(df, title='Raw Forest train data profiling report', explorative=True)
    profiling.to_file(report_path)