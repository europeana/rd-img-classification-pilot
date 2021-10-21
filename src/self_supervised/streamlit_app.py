
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

import fire

def main(**kwargs):

    img_path =  '/home/jcejudo/projects/image_classification/data/single_label/images_training/photograph/[ph]2021653[ph]https___www_archieval_nl_detail_php_id_1355014.jpg'

    img = Image.open(img_path)

    #st.button("add" , img=img,  bold=True)
    st.image(img)

    # df_path = kwargs.get('df_path')
    # df_path = '/home/jcejudo/rd-img-classification-pilot/data/multilabel/multilabel_dataset_open_permission.csv'
    # df = pd.read_csv(df_path)

    # unique_categories = []
    # for cat in df['category'].values:
    #     unique_categories += cat.split()
        
    # unique_categories = list(set(unique_categories))

    # st.title('Europeana image classification pilot')

    # #to do: show count of different categories 

    # st.sidebar.markdown('# Choose the categories')

    # n_results = st.sidebar.text_input("number of results", '200')
    # n_results = int(n_results)

    # choices = st.sidebar.multiselect('categories',unique_categories)
    # cat_to_display = list(choices)

    # if len(cat_to_display) == 1:
    #     _df = df.loc[df['category'].apply(lambda x: cat_to_display[0] in x)]
    # else:
    #     _df = df.loc[df['category'].apply(lambda x: set(cat_to_display) == set(x.split()))]

    # id_list = list(_df['ID'].values)[:n_results]
    # url_list = list(_df['URL'].values)[:n_results]

    # id_link_list = [f'http://data.europeana.eu/item{id}' for id in id_list]

    # st.image(url_list,width=300,caption = id_link_list)



if __name__ == '__main__':

    fire.Fire(main)