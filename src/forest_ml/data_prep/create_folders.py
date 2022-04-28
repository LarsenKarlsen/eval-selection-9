from os import path, getcwd, mkdir


def create_data_folders():
    """
    Creates data dir with necessary folders
    if data dir with necessary folders exists return True
    """
    data_path = path.join('..', getcwd(), 'data')
    processed_path = path.join('..', getcwd(), 'data', 'processed')
    raw_path = path.join('..', getcwd(), 'data', 'raw')

    if path.exists(data_path)==False: # see if we have our data dir if not create
        mkdir(data_path)
        mkdir(processed_path)
        mkdir(raw_path)
    if path.exists(processed_path)==False: # see if data dir have processed dir if not create it
        mkdir(processed_path)
    if path.exists(raw_path)==False: # see if data dir have processed dir if not create it
        mkdir(raw_path)
    
    if path.exists(processed_path) and path.exists(raw_path):
        return True
    
create_data_folders()