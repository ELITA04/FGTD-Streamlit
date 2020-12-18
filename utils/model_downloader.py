import os
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_models():
	if not os.path.exists('./models'):
		gdd.download_file_from_google_drive(file_id='1bd6wUZhdJoMD4t49-K7p2Q0hhp9gNs8g',
                                    dest_path='./models.zip',
                                    unzip=True)
		os.remove('./models.zip')