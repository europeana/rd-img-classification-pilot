#download images
import os
import requests
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


if __name__ == "__main__":

    ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    dest_path = os.path.join(ROOT_DIR,'training_data')
    create_dir(dest_path)
    
    df = pd.read_csv(os.path.join(ROOT_DIR,'dataset.csv'))
    
    for cat in df.category.unique():
        print(cat)
        #subset 
        df_category = df.loc[df['category'] == cat]
        
        cat_path = os.path.join(dest_path,cat)
        create_dir(cat_path)
        
        for i in range(df_category.shape[0]):
            ID = df_category['ID'].iloc[i]
            img = url2img(df_category['URL'].iloc[i])
            
            if img:
                try:
                    img.save(os.path.join(cat_path,f'{ID}.jpg'.replace("/","[ph]")))
                except:
                    pass

    
    