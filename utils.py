import os
import urllib.request

from os import makedirs
from os.path import join as join_path, exists
from zipfile import ZipFile

def download_csv(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

