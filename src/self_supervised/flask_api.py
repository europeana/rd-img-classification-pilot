
from flask import Flask, jsonify, request
import sys
sys.path.append('../')

import lightly.utils.io as io
import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import json
import argparse

import pandas as pd
from pathlib import Path

from flask import session


import fire

def main(**kwargs):
    embeddings_path = kwargs.get('embeddings_path')
    data_path = kwargs.get('data_path')
    n_neighbors = kwargs.get('n_neighbors',9)
    endpoint_name = kwargs.get('endpoint_name','img_recommendation')
    host = kwargs.get('host','0.0.0.0')
    port = kwargs.get('port',5000)

    # load dataframe with provenance
    df_path = '/home/jcejudo/projects/image_classification/data/single_label/training_data.csv'

    print('Loading provenance df')
    provenance_df = pd.read_csv(df_path)

    #provenance_df = provenance_df.loc[provenance_df['ID'].apply(lambda x: x in embedded_id_list)]

    print('Loading embeddings')

    #load embeddings
    data_path = Path(data_path)
    df = pd.read_csv(embeddings_path)
    df = df.drop_duplicates(subset='filenames', keep="first")

    # to do: calculate id

    def filename_to_id(fname):
        fname = Path(fname).with_suffix('')
        fname = str(fname).split('/')[1]
        fname = fname.replace('[ph]','/')
        return fname

    df['ID'] = df['filenames'].apply(filename_to_id)

    
    df = df.merge(provenance_df, left_on='ID', right_on='ID')

    df = df.drop_duplicates(subset='filenames', keep="first")

    

    

    embeddings = df[[c for c in df.columns if 'embedding' in c]].values
    embeddings = normalize(embeddings)
    filenames = list(df['filenames'].values)
    urls = list(df['URL'].values)
    uris = list(df['URI'].values)

    #print('Embeddings loaded')

    #print('Getting IDs')

    # # get ids
    # embedded_id_list = []
    # for fname in filenames:
    #     fname = Path(fname).with_suffix('')
    #     fname = str(fname).split('/')[1]
    #     fname = fname.replace('[ph]','/')
    #     embedded_id_list.append(fname)

    # print('Loading provenance dataframe')
    
    # print(provenance_df.shape)
    # print(len(embedded_id_list))

    # print('Getting urls and uris')

    # url_list = []
    # uri_list = []
    # for id in embedded_id_list:
    #     matches = provenance_df.loc[provenance_df['ID'].apply(lambda x: x == id)]
    #     url = matches['URL'].values[0]
    #     uri = matches['URI'].values[0]
    #     url_list.append(url)
    #     uri_list.append(uri)

    

    print('Finished loading data')


    def get_data():

        return {
            'filenames':filenames,
            'urls':urls,
            'uris':uris,

        }




    # to do: match ids with provenance

    def get_filepaths():
        return [str(data_path.joinpath(fname)) for fname in filenames]

    def get_filenames():
        return filenames

    app = Flask(__name__)
        
    @app.route(f'/{endpoint_name}', methods=['POST','GET'])
    def predict():
        if request.method == 'GET':
            data = get_data()
            
            data.update({'fpaths':[str(data_path.joinpath(fname)) for fname in data['filenames']]})
            #filenames = get_filepaths()
            return jsonify(data)

        elif request.method == 'POST':

            output_dict = {
                'fpaths':[],
                'uris':[],
                'urls':[],
                }

            file = None
            for k in request.files.keys():
                file = request.files[k]
            
            if file:
                img_bytes = file.read()   
                pred_filename = file.filename

                data = get_data()

                filenames = data['filenames']
                uris = data['uris']
                urls = data['urls']

                matches = [s for s in filenames if pred_filename in s]
                if matches:

                    match = matches[0]
                    example_idx = filenames.index(match)
                    # get distances to the cluster center
                    distances = embeddings - embeddings[example_idx]
                    distances = np.power(distances, 2).sum(-1).squeeze()
                    # sort indices by distance to the center
                    nearest_neighbors = np.argsort(distances)[:n_neighbors]

                    for idx in nearest_neighbors:
                        fname = os.path.join(data_path, filenames[idx])
                        fname = Path(fname)
                        if fname.is_file():
                            output_dict['fpaths'].append(str(fname))
                            output_dict['uris'].append(uris[idx])
                            output_dict['urls'].append(urls[idx])
                
            return jsonify(output_dict)


    app.run(host=host, port=port)
    return 



if __name__=="__main__":
    fire.Fire(main)




