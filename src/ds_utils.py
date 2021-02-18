import torch
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def path2DataFrame(data_dir):
    """reads a directory structured in classes into a dataframe"""
    category_list = os.listdir(data_dir)
    img_path_list = []
    img_category_list = []
    for cat in category_list:
        cat_path = os.path.join(data_dir,cat)
        img_cat_list = [os.path.join(cat_path,filename) for filename in os.listdir(cat_path)]
        img_path_list += img_cat_list
        img_category_list += [cat for i in range(len(img_cat_list))]
        
    return pd.DataFrame({'file_path':img_path_list,'category':img_category_list})

def drop_categories_df(df,categories):
    """categories can be str or list of str"""
    if isinstance(categories,str):
        categories = [categories]
    def filter(x):
        return x not in categories
    # apply filter
    df_ = df[df['category'].map(filter)]
    # re-index the dataframe
    df_.index = np.arange(df_.shape[0])
    return df_







