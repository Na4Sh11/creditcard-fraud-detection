import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    os.makedirs("data", exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.dataset_download_file(
        dataset='mlg-ulb/creditcardfraud',
        file_name='creditcard.csv',
        path='data'
    )
    print("Download complete!")

if __name__ == '__main__':
    download_dataset()
