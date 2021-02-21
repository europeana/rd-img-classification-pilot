#download images
import os
#import requests
import argparse
from PIL import Image
import pandas as pd
from io import BytesIO
from ds_utils import create_dir
import urllib
from multiprocessing import Process
import time

#to do: https://stackoverflow.com/questions/62517121/how-to-use-multiprocessing-to-download-images-using-requests


def url2img(url):
    try:
        return Image.open(urllib.request.urlopen(url)).convert('RGB')
    except:
        print('Failed to get media image')
        return None

def download_single_image(url,ID,saving_dir):
  img = url2img(url)
  ID = ID.replace("/","[ph]")
  fname = f'{ID}.jpg'
  if img:
      try:
          img.save(os.path.join(saving_dir,fname))
      except:
          pass

def download_images(csv_path,saving_dir):
    
    time_limit = 7

    create_dir(saving_dir)
    df = pd.read_csv(csv_path)
    
    for cat in df.category.unique():
        print(cat)
        df_category = df.loc[df['category'] == cat]
        cat_path = os.path.join(saving_dir,cat)
        create_dir(cat_path)

        #to do: use .value instead of looping the indexes
        
        for i in range(df_category.shape[0]):
            ID = df_category['ID'].iloc[i]
            URL = df_category['URL'].iloc[i]

            action_process = Process(target=download_single_image,args=(URL,ID,cat_path))
            action_process.start()
            action_process.join(timeout=time_limit) 
            action_process.terminate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--saving_dir', required=True)
    args = parser.parse_args()

    download_images(args.csv_path,args.saving_dir)





    
    