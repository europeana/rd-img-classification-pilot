#download images
import os
import requests
from PIL import Image
import pandas as pd
from io import BytesIO

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def url2img(url):
    try:
        response = requests.get(url)
        return Image.open(BytesIO(response.content)).convert('RGB')
    except:
        print('Failed to get media image')
        pass

data = 'ec'
data_path = os.path.join('/home/jcejudo/rd-img-classification-pilot/data_sample',data)
dest_path = '/home/jcejudo/rd-img-classification-pilot/training_sample'
create_dir(dest_path)
dest_path = os.path.join(dest_path,data)
create_dir(dest_path)

exclude_cat = os.listdir(dest_path)

if data == 'getty':
    exclude_cat += ['print','tapestry','drawing','tool','jewellery','photograph','poster','painting','ceramics','textile','specimen','woodwork','machinery','building','furniture','cartoon','toy','map','postcard','sculpture','food','glassware','metalwork','medal','memorabilia','mineral','musical_instrument','tableware','stamp']

for cat in os.listdir(data_path):

    if cat.replace('.csv','') in exclude_cat:
        continue

    print(cat)
    
    cat_path = os.path.join(data_path,cat)
    
    df = pd.read_csv(cat_path)
    
    # if cat.replace('.csv','') in ['ceramics_data','map_data']:
    #     continue
    
    dest_cat_path = os.path.join(dest_path,cat.replace('.csv',''))
    create_dir(dest_cat_path)
    for i in range(df.shape[0]):
        ID = df['ID'].iloc[i]
        url = df['URL'].iloc[i]
        img = url2img(url)
        
        #replacement = "%2F"
        replacement = "__placeholder__"
        filename = f'{ID}.jpg'.replace("/",replacement)
        if img:
            try:
                img.save(os.path.join(dest_cat_path,filename))
            except:
                pass
    
    