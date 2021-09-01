import os
import argparse
from PIL import Image
import pandas as pd
from io import BytesIO
from ds_utils import create_dir
import urllib
from multiprocessing import Process
import time
from tqdm import tqdm

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

def download_single_label_dataset(csv_path,saving_dir,mode):

    time_limit = 7

    create_dir(saving_dir)
    df = pd.read_csv(csv_path)
    df = df[['URI', 'ID', 'URL', 'category']]
    df = df.dropna()
    
    for cat in df.category.unique():
        print(cat)
        df_category = df.loc[df['category'] == cat]
        cat_path = os.path.join(saving_dir,cat)
        create_dir(cat_path)

        for ID,URL in zip(df_category['ID'].values,df_category['URL'].values):
        
            action_process = Process(target=download_single_image,args=(URL,ID,cat_path))
            action_process.start()
            action_process.join(timeout=time_limit) 
            action_process.terminate()

def download_multi_label_dataset(csv_path,saving_dir,mode):

    time_limit = 7

    create_dir(saving_dir)
    df = pd.read_csv(csv_path)
    print('number of images for downloading: '+str(df.shape[0]))

    # for cat in df.category.unique():
    #     print(cat)
    #     df_category = df.loc[df['category'] == cat]
    #     cat_path = os.path.join(saving_dir,cat)
    #     create_dir(cat_path)

    for ID,URL in tqdm(zip(df['ID'].values,df['URL'].values)):
    
        action_process = Process(target=download_single_image,args=(URL,ID,saving_dir))
        action_process.start()
        action_process.join(timeout=time_limit) 
        action_process.terminate()

def download_images(csv_path,saving_dir,mode):

    if not mode:
        mode = 'single_label'

    if mode == 'single_label':
        download_single_label_dataset(csv_path,saving_dir,mode)
    elif mode == 'multi_label':
        download_multi_label_dataset(csv_path,saving_dir,mode)



if __name__ == "__main__":

    """
    Script for downloading the images of the dataset, 
    which will be stored in directories corresponding to the different categories

    Usage:

      python src/download_images.py --csv_path dataset_3000.csv --saving_dir training_data

    Parameters:

      csv_path: csv file obtained by the script 'harvest_data.py'
                  Required

      saving_dir: directory for saving the images. 
                  Required

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--saving_dir', required=True)
    parser.add_argument('--mode', required=False)
    args = parser.parse_args()

    download_images(args.csv_path,args.saving_dir,args.mode)





    
    