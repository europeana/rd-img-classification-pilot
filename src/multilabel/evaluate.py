import argparse
import os
import fire
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
import torch.optim as optim

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
import sys
sys.path.append(os.path.join(ROOT_DIR))

from gradcam import *
from torch_utils import *
from train_multilabel import *

def main(**kwargs):
  annotations = kwargs.get('annotations')
  data_dir = kwargs.get('data_dir')
  saving_dir = kwargs.get('saving_dir')
  input_size = kwargs.get('input_size')
  batch_size = kwargs.get('batch_size',32)
  num_workers = kwargs.get('num_workers',4)

  results_path = kwargs.get('results_path')

  threshold = kwargs.get('threshold',0.5)
  average = kwargs.get('average','micro')

  data_dir = Path(data_dir)
  df_path = Path(annotations)
  saving_dir = Path(saving_dir)
  saving_dir.mkdir(parents=True, exist_ok=True)

  metrics_list = ['acc','precision','recall','f1','coverage','lrap','label_ranking_loss','ndcg_score','dcg_score']

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  df = pd.read_csv(df_path)
  #df = df.dropna()
  #filter images in df contained in data_path
  imgs_list = list(data_dir.iterdir())
  df['filepath'] = df['ID'].apply(lambda x:data_dir.joinpath(id_to_filename(x)+'.jpg'))
  df = df.loc[df['filepath'].apply(lambda x: Path(x) in imgs_list)]
  

  imgs = np.array([str(path) for path in df['filepath'].values])
  labels = [item.split() for item in df['category'].values]


  metrics_dict = {m:[] for m in metrics_list}


  results_path = Path(results_path)

  # get list of files
  file_list = [x for x in results_path.iterdir() if not x.is_dir()]
  split_list = [x for x in results_path.iterdir() if x.is_dir()]

  cat_metrics = None

  if file_list:
    split_list = [results_path]

  for split_path in split_list:
    
    conf_path = split_path.joinpath('conf.json')

    with open(conf_path,'r') as f:
      conf = json.load(f)

    input_size = conf['input_size']
    resnet_size = conf['resnet_size']

    model,class_index_dict = load_multilabel_model(split_path,device,resnet_size = resnet_size)

    

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    mlb = sklearn.preprocessing.MultiLabelBinarizer()
    mlb.fit([class_index_dict.values()])

    labels = mlb.transform(labels)

    testset = MultilabelDataset(imgs,labels,transform = test_transform)
    testloader = DataLoader(testset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

    print('test:',imgs.shape[0])

    loss_function = nn.BCEWithLogitsLoss()

    test_metrics = evaluate(
        model = model,
        dataloader = testloader,
        loss_function = loss_function,
        device = device,
        threshold = threshold,
        class_index_dict = class_index_dict,
        average = average
    )

    #print(test_metrics['confusion_matrix'])

    

    if cat_metrics is None:
      cat_metrics = {k:{'precision':[],'recall':[],'f1-score':[],} for k in class_index_dict.values()}

    cm_dict = {}
    for category, cm in zip(cat_metrics.keys(),test_metrics['confusion_matrix']):

      cm_dict.update({category:cm})

      # to do: calculate precision, recall and f1 from confusion matrix here

      TN = float(cm[0,0])
      TP = float(cm[1,1])
      FP = float(cm[0,1])
      FN = float(cm[1,0])

      try:
          recall = TP/(TP+FN)
      except:
          recall = 0
      try:
          precision = TP/(TP+FP)
      except:
          precision = 0
          
      try:
          f1 = 2*TP/(2*TP+FP+FN)
      except:
          f1 = 0

      cat_metrics[category]['precision'].append(precision)
      cat_metrics[category]['recall'].append(recall)
      cat_metrics[category]['f1-score'].append(f1)
      
      #cm_dict[category] += cm


    #print_metrics(test_metrics)

    for metrics in metrics_list:
      #print(metrics,test_metrics[metrics])
      metrics_dict[metrics].append(test_metrics[metrics])

  for m in metrics_list:
    if m not in ['confusion_matrix']:
      metrics_dict[m] = {'mean':np.mean(metrics_dict[m]),'std':np.std(metrics_dict[m])}

  for m in metrics_list:
    if m not in ['confusion_matrix']:
      print(m,metrics_dict[m]['mean'],'+-',metrics_dict[m]['std'] )

  for k in cat_metrics.keys():
      cat_metrics[k]['precision'] = {'mean':np.mean(cat_metrics[k]['precision']),'std':np.std(cat_metrics[k]['precision'])}
      cat_metrics[k]['recall'] = {'mean':np.mean(cat_metrics[k]['recall']),'std':np.std(cat_metrics[k]['recall'])}
      cat_metrics[k]['f1-score'] = {'mean':np.mean(cat_metrics[k]['f1-score']),'std':np.std(cat_metrics[k]['f1-score'])}

  # for category in cm_dict.keys():
  #   cm_dict[category] = np.round(cm_dict[category]/len(split_list))

  metrics_dict.update({'report':cat_metrics})
  metrics_dict.update({'confusion_matrix':cm_dict})

  torch.save(metrics_dict,saving_dir.joinpath('evaluation_results.pth'))

  #recall = metrics['recall']
  
  #print(f'loss:{loss:.3f} acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} recall:{recall:.3f}')
  # save_json(test_metrics,saving_dir.joinpath('test_metrics.json'))


if __name__ == '__main__':
    fire.Fire(main)





        












