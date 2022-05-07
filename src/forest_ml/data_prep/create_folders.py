from os import path, getcwd, mkdir
import click


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
        click.echo(f'[+] Data dir with apropriate catalogs created')
    if path.exists(processed_path)==False: # see if data dir have processed dir if not create it
        mkdir(processed_path)
        click.echo(f'[+] Add processed dir to data')
    if path.exists(raw_path)==False: # see if data dir have processed dir if not create it
        mkdir(raw_path)
        click.echo(f'[+] Add raw dir to data')
    
    
    if path.exists(processed_path) and path.exists(raw_path):
        click.echo(f'[+] Data dir with apropriate catalog exists')
        return True
    

def create_models_folder():
    """
    Create folder to store trained models
    """
    data_path = path.join('..', getcwd(), 'models')
    if path.exists(data_path)==False: # see if we have our models dir if not create
        mkdir(data_path)
        click.echo(f'[+] Created models dir in {data_path}')
    else:
        click.echo(f'[+] Models dir exists')
    return True

def  create_notebook_folder():
    """
    Create notebooks dir
    """
    report_path = path.join('..', getcwd(), 'notebooks')
    if path.exists(report_path)==False:
        mkdir(report_path)
        click.echo(f'[+] Created reports dir at {report_path}')
    else:
        click.echo(f'[+] Report dir exist')
    return True