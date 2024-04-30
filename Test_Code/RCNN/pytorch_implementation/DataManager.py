from pathlib import Path
import gdown
import os
import urllib
import shutil
from urllib.request import urlopen, urlretrieve
import sys
import zipfile

class DataManager:
    
    # Dictionary of dataset names and download URLs
    _urls = {
        'train' : ['images_thermal_train.zip', 'https://drive.google.com/uc?id=1iJb_4R3tpyMO9Qd0aTWs0l6t18FuO3mR'],
        'val'   : ['images_thermal_val.zip',   'https://drive.google.com/uc?id=1CNcXLTOFKLAxuj81PGdhB0XfuDyOemAu'],
        'test'  : ['video_thermal_test.zip',   'https://drive.google.com/uc?id=1COJPkI72k3wC5GqRwG_d8tK_p_9LdBz8']}

    # URL to full download directory
    _full_download_url = 'https://adas-dataset-v2.flirconservator.com/dataset/full/FLIR_ADAS_v2.zip'

    # Class constructor
    def __init__(self, datasets='all'):

        # Ensure dataset is a list
        if datasets is not list:
            datasets = [datasets]

        # Select all valid datasets if 'all' is provided
        if any([dataset == 'all' for dataset in datasets]):
            datasets = self._urls.keys()

        # Ensure that all datasets are valid
        for dataset in datasets:
            if dataset not in self._urls.keys():
                raise ValueError('Invalid dataset %s' % (dataset))
            
        # Save dataset to datastore
        self.datasets = datasets

    # Determine download directory
    def get_download_dir(self):
        home_dir = Path.home()
        download_dir = os.path.join(home_dir,'FLIR_ADAS_v2')
        return download_dir
    
    # Download datasets
    def download_datasets(self):
        
        # Get the path to the download directory
        download_dir = self.get_download_dir()

        # Download error
        download_error = False

        # Loop for each dataset
        for dataset in self.datasets:

            # Get path to dataset directory
            zip_file = self._urls[dataset][0]
            zip_file = os.path.join(download_dir, zip_file)
            dataset_dir = os.path.splitext(zip_file)[0]

            # Determine if dataset directory already exists
            if os.path.exists(dataset_dir):
                print("'%s' dataset exists. Bypassing download..." % (dataset))

            # Download dataset directory if it does not exist
            else:
                try:
                    if not os.path.exists(download_dir):
                        os.makedirs(download_dir)
                    gdown.download(self._urls[dataset][1], output=zip_file)
                    with zipfile.ZipFile(zip_file, 'r') as z:
                        z.extractall(download_dir)
                    os.remove(zip_file)
                except:
                    download_error = True
                    break

        # If download from google drive failed
        if download_error:

            # If download failed due to no internet connection
            if not self.has_internet_connection():
                raise Exception('No internet connection. Unable to download datasets')
            
            # If download failed because links were invalid
            else:

                # Alert user that full dataset will be downloaded
                print('Warning: Unable to Access Google Drive Links. Downloading until FLIR_ADAS_v2 dataset...')
                while True:
                    continue_download = input('Are you sure you want to continue (Y/N)? ')
                    if continue_download.upper() == 'N':
                        sys.exit(0)
                    elif continue_download.upper() == 'Y':
                        break
                    else:
                        print('Invalid input. Please try again.')

                # Determine if the download location already exists
                if os.path.exists(download_dir):

                    # If empty download location exists, delete it
                    if len(os.listdir(download_dir)) == 0:
                        os.rmdir(download_dir)

                    # If non-empty download directory exists
                    else:

                        # Alert user and ensure they are okay with deleting it
                        print('Warning: Non-empty FLIR_ADAS_v2 directory already exists. This directory will be deleted.')
                        while True:
                            continue_download = input('Are you sure you want to continue (Y/N)? ')
                            if continue_download.upper() == 'N':
                                sys.exit(0)
                            elif continue_download.upper() == 'Y':
                                break
                            else:
                                print('Invalid input. Please try again.')

                        # Remove download directory
                        shutil.rmtree(download_dir)

                # Download full FLIR ADAS v2 dataset
                parent_dir = os.path.abspath(os.path.join(download_dir,'..'))
                zip_file = os.path.join(parent_dir,'FLIR_ADAS_v2.zip')
                urlretrieve(self._full_download_url, zip_file)

                # Unzip full dataset
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(parent_dir)
                os.remove(zip_file)

    def has_internet_connection(_):
        try:
            urlopen('https://www.google.com', timeout=1)
            return True
        except urllib.error.URLError as Error:
            return False
        
if __name__ == "__main__":
    data_manager = DataManager('val')
    data_manager.download_datasets()

# gdown.download('https://drive.google.com/uc?id=1CNcXLTOFKLAxuj81PGdhB0XfuDyOemAu')
# url = "https://drive.google.com/file/d/1CNcXLTOFKLAxuj81PGdhB0XfuDyOemAu/view?usp=drive_link"
# url = "https://drive.google.com/uc?export=download&id=1CNcXLTOFKLAxuj81PGdhB0XfuDyOemAu"
# filename = "images_thermal_val.zip"
# urlretrieve(url,filename)#,show_progress)