import os
import zipfile
from fastdownload import FastDownload
# from torch.hub import download_url_to_file


def unzip(source_item, dest_dir):
    with zipfile.ZipFile(source_item) as zf:
        zf.extractall(path=dest_dir)
    

if __name__ == '__main__':
    # _download_url_to_file('https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1', 'saved_models.zip', None, True)
    # unzip('saved_models.zip', '.')
    d = FastDownload(base='./media/models', archive='downloaded', data='extracted')
    d.get('https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1')        # change to my target url