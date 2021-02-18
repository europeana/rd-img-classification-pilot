#download images
import os
import requests
import argparse
from PIL import Image
import pandas as pd
from io import BytesIO
from ds_utils import create_dir

def url2img(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert('RGB')
    except:
        print('Failed to get media image')
        pass

def download_main(csv_path,saving_dir):

    create_dir(saving_dir)
    df = pd.read_csv(csv_path)
    
    for cat in df.category.unique():
        print(cat)
        #subset 
        df_category = df.loc[df['category'] == cat]
        
        cat_path = os.path.join(saving_dir,cat)
        create_dir(cat_path)
        
        for i in range(df_category.shape[0]):
            ID = df_category['ID'].iloc[i]
            img = url2img(df_category['URL'].iloc[i])
            
            if img:
                try:
                    img.save(os.path.join(cat_path,f'{ID}.jpg'.replace("/","[ph]")))
                except:
                    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--saving_dir', required=True)
    parser.add_argument('--csv_path', required=True)
    args = parser.parse_args()

    download_main(args.csv_path,args.saving_dir)





    
    