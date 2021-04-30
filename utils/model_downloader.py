import os
from google_drive_downloader import GoogleDriveDownloader as gdd


def download_models():
    if not os.path.exists("./models"):
        gdd.download_file_from_google_drive(
            file_id="1WxgfF1R9R_aaKnYVdEaEvdOHiEWwHjjo",
            dest_path="./models.zip",
            showsize=True,
            unzip=True,
        )
        os.remove("./models.zip")
