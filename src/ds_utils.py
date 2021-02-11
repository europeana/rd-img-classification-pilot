from notebook_env import *

import torch
import os
import pandas as pd

#to do: document

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

def get_class_weights(y_encoded,encoding_dict):
    """Calculates the weights for the Cross Entropy loss """
    data_dict = get_imgs_per_cat(y_encoded)       
    N = sum(data_dict.values())
    #calculate weights as the inverse of the frequency of each class
    weights = []
    for k in data_dict.keys(): 
        weights.append(N/data_dict[k])
    return weights

def get_imgs_per_cat(y_encoded):
    #count the images in each category
    data_dict = {}
    for el in y_encoded:
        if el not in data_dict.keys():
            data_dict.update({el:1})
        else:
            data_dict[el] += 1
    return data_dict

def label_encoding(y):
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)
    encoding_dict = {}
    for cat in le.classes_:
        label = le.transform(np.array([cat]))[0]
        encoding_dict.update({int(label):cat}) 
    return y_encoded, encoding_dict

class Experiment():
    def __init__(self):
        self.info = {}

    def add(self,key,value):
        self.info.update({key:value})
        return self

    def show(self):
        print(f'keys: {self.info.keys()}\n')
        for k,v in self.info.items():
            print(f'{k}: {v}\n')

    def save(self,dest_path):
        filename = 'training_info.pth'
        info_file_path = os.path.join(dest_path,filename)
        torch.save(self.info, info_file_path)






