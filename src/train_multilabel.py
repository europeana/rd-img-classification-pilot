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

def id_to_filename(id):
    if not isinstance(id,str):
      id = str(id)
    return id.replace('/','[ph]')

class MultilabelResNet(nn.Module):
    def __init__(self,size, output_size):
        super(MultilabelResNet, self).__init__()

        if size not in [18,34,50,101,152]:
            raise Exception('Wrong size for resnet')
        if size == 18:
            self.net = torchvision.models.resnet18(pretrained=True)
        elif size == 34:
            self.net = torchvision.models.resnet34(pretrained=True)
        elif size == 50:
            self.net = torchvision.models.resnet50(pretrained=True)
        elif size == 101:
            self.net = torchvision.models.resnet101(pretrained=True)
        elif size == 152:
            self.net = torchvision.models.resnet152(pretrained=True)

        #initialize the fully connected layer
        self.net.fc = nn.Linear(self.net.fc.in_features, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.net(x)
        out = self.sigmoid(out)
        return out
    
class MultilabelDataset(Dataset):
    """
    Pytorch training dataset class
    X: Numpy array containing the paths to the images
    y: Numpy array with the encoded labels
    returns batches of images, labels and images paths
    """
    def __init__(self, X, y, transform=None):
        self.transform = transform
        self.X = X
        self.y = y        

    def __len__(self):
        return self.y.shape[0] 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.X[idx]
        img, label = Image.open(img_path).convert('RGB'), self.y[idx]
        if self.transform:
            img = self.transform(img)

        return img,label,img_path


def load_multilabel_model(root_path, device, resnet_size = 18):

    with open(os.path.join(root_path,'class_index.json'),'r') as f:
        class_index_dict = json.load(f)

    class_index_dict = {int(k):v for k,v in class_index_dict.items()}

    checkpoint_path = os.path.join(root_path,'checkpoint.pth')

    model = MultilabelResNet(resnet_size,len(class_index_dict)).to(device)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(checkpoint_path))

    model = model.to(device)
    model.eval()

    return model, class_index_dict


def print_metrics(metrics):
    loss = metrics['loss']
    coverage = metrics['coverage']
    lrap = metrics['lrap']
    label_ranking_loss = metrics['label_ranking_loss']
    ndcg_score = metrics['ndcg_score']
    dcg_score = metrics['dcg_score']
    #print(f'loss:{loss:.3f} coverage:{coverage:.3f} lrap:{lrap:.3f} label_ranking_loss:{label_ranking_loss:.3f} ndcg_score:{ndcg_score:.3f} dcg_score:{dcg_score:.3f}')

    acc = metrics['acc']
    f1 = metrics['f1']
    precision = metrics['precision']
    recall = metrics['recall']
    
    print(f'loss:{loss:.3f} acc:{acc:.3f} f1:{f1:.3f} precision:{precision:.3f} recall:{recall:.3f}')

def evaluate(**kwargs):
    
    model = kwargs.get('model')
    dataloader = kwargs.get('dataloader')
    loss_function = kwargs.get('loss_function')
    device = kwargs.get('device')
    average = kwargs.get('average','macro')
    thres = kwargs.get('thres',0.8)
    
    ground_truth = []
    predictions = []
    
    model.eval()
    val_loss = 0.0
    for inputs,labels,_ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        labels = labels.type_as(output)
        loss = loss_function(output, labels)
        val_loss += loss.item()
                
        ground_truth += list(labels.cpu().detach().numpy())
        predictions += list(output.cpu().detach().numpy())
        
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
        
    val_loss /= len(dataloader.dataset)
    coverage = sklearn.metrics.coverage_error(ground_truth, predictions)
    lrap = sklearn.metrics.label_ranking_average_precision_score(ground_truth, predictions)
    label_ranking_loss = sklearn.metrics.label_ranking_loss(ground_truth, predictions)
    ndcg_score = sklearn.metrics.ndcg_score(ground_truth, predictions)
    dcg_score = sklearn.metrics.dcg_score(ground_truth, predictions)

    y_hard_pred = np.where(predictions > thres, 1, 0)

    acc = sklearn.metrics.accuracy_score(ground_truth, y_hard_pred)
    recall = sklearn.metrics.recall_score(ground_truth, y_hard_pred,average=average)
    precision = sklearn.metrics.precision_score(ground_truth, y_hard_pred,average=average)
    f1 = sklearn.metrics.f1_score(ground_truth, y_hard_pred,average=average)

    cm_arr = sklearn.metrics.multilabel_confusion_matrix(ground_truth, y_hard_pred)

    return {
        'loss':val_loss,
        'coverage':coverage,
        'lrap':lrap,
        'label_ranking_loss':label_ranking_loss,
        'ndcg_score':ndcg_score,
        'dcg_score':dcg_score,
        'acc':acc,
        'recall':recall,
        'precision':precision,
        'f1':f1,
        'confusion_matrix':cm_arr
        }

def save_json(metrics,path):
  with open(path,'w') as f:
    json.dump(metrics,f)

def train(**kwargs):
    model = kwargs.get('model')
    trainloader = kwargs.get('trainloader')
    valloader = kwargs.get('valloader')
    device = kwargs.get('device')
    optimizer = kwargs.get('optimizer')
    scheduler = kwargs.get('scheduler')
    loss_function = kwargs.get('loss_function')
    saving_dir = kwargs.get('saving_dir')
    max_epochs = kwargs.get('max_epochs',100)
    patience = kwargs.get('patience',10)

    model.train()

    count = 0
    best_loss = 1e9
    for epoch in range(max_epochs):
        train_loss = 0.0

        
        # loop over batches
        for i, (inputs,labels,_) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()        
            output = model(inputs)
            labels = labels.type_as(output)
            loss = loss_function(output, labels)
            #backpropagate and update
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(trainloader.dataset)

        val_metrics = evaluate(
            model = model,
            dataloader = valloader,
            loss_function = loss_function,
            device = device
        )
        print_metrics(val_metrics)

        if scheduler:
          scheduler.step()
        
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            best_metrics = val_metrics
            count = 0
            torch.save(model.state_dict(),saving_dir.joinpath(f'checkpoint.pth'))
        else:
            count += 1
        if count > patience:
            break

    model.load_state_dict(torch.load(saving_dir.joinpath(f'checkpoint.pth')))

    return model

def drop_categories(df,cat_list):
    if isinstance(cat_list,str):
        cat_list = [cat_list]
    
    _df = df.copy()
    for cat in cat_list:
        _df = _df.loc[_df['category'].apply(lambda x: cat not in x)]
    return _df

def calculate_weights(df,class_index_dict):
  unique_categories = []
  for cat in df['category'].values:
      unique_categories += cat.split()
      
  unique_categories = list(set(unique_categories))

  occurrence_dict = {}
  for cat in unique_categories:
      occurrence_dict.update({cat:df.loc[df['category'].apply(lambda x: cat in x)].shape[0]})
  total = sum(occurrence_dict.values())
  weights_dict = {k:total/v for k,v in occurrence_dict.items()}

  weights_list = []
  for i in range(len(class_index_dict)):
    weights_list.append(weights_dict[class_index_dict[i]])

  weights_torch = torch.tensor(weights_list)
  return weights_torch


def main(**kwargs):
    max_epochs = kwargs.get('max_epochs',100)
    annotations = kwargs.get('annotations')
    data_dir = kwargs.get('data_dir')
    saving_dir = kwargs.get('saving_dir')
    input_size = kwargs.get('input_size',128)
    batch_size = kwargs.get('batch_size',32)
    num_workers = kwargs.get('num_workers',4)
    learning_rate = kwargs.get('learning_rate',1e-4)
    sample = kwargs.get('sample',1.0)
    crossvalidation = kwargs.get('crossvalidation',False)
    resnet_size = kwargs.get('resnet_size',18)
    n_splits = kwargs.get('n_splits',10)
    hf_prob = kwargs.get('hf_prob',0.5)
    vf_prob = kwargs.get('vf_prob',0.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = Path(data_dir)
    df_path = Path(annotations)
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)

    conf = {
        'resnet_size':resnet_size,
        'input_size':input_size
    }

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(hf_prob),
        transforms.RandomVerticalFlip(vf_prob),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(df_path)
    df = df.groupby('category').apply(lambda x: x.sample(frac=sample))

    #filter images in df contained in data_path
    imgs_list = list(data_dir.iterdir())
    df['filepath'] = df['ID'].apply(lambda x:data_dir.joinpath(id_to_filename(x)+'.jpg'))
    df = df.loc[df['filepath'].apply(lambda x: Path(x) in imgs_list)]

    df['n_labels'] = df['category'].apply(lambda x: len(x.split()))
    df = df.sort_values(by='n_labels',ascending=False)
    df = df.drop_duplicates(keep='first',subset='ID')
    df = drop_categories(df,['specimen','clothing'])
    print(df.shape)

    mlb = sklearn.preprocessing.MultiLabelBinarizer()

    imgs = np.array([str(path) for path in df['filepath'].values])

    labels = [item.split() for item in df['category'].values]
    labels = mlb.fit_transform(labels)

    class_index_dict = {i:c for i,c in enumerate(mlb.classes_)}
    print(class_index_dict)



    skf = KFold(n_splits=n_splits)
    sk_splits = skf.split(imgs, labels)
    splits_list = []
    for i,(train_val_index, test_index) in enumerate(sk_splits):

        if not crossvalidation and i>0:
            break

        print(f'split_{i}')

        X_train_val, X_test = imgs[train_val_index], imgs[test_index]
        y_train_val, y_test = labels[train_val_index], labels[test_index]
        
        val_sk_splits = skf.split(X_train_val, y_train_val)
        train_index,val_index = next(val_sk_splits)
        
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        split_path = saving_dir.joinpath(f'split_{i}')
        split_path.mkdir(parents=True, exist_ok=True)

        conf_path = split_path.joinpath('conf.json')
        with open(conf_path,'w') as f:
            json.dump(conf,f)

        #save class index dict
        save_json(class_index_dict,split_path.joinpath('class_index.json'))

        trainset = MultilabelDataset(X_train,y_train,transform = train_transform)
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True
            )

        valset = MultilabelDataset(X_val,y_val,transform = test_transform)
        valloader = DataLoader(
            valset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True
            )

        testset = MultilabelDataset(X_test,y_test,transform = test_transform)
        testloader = DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True
            )

        print('train:',X_train.shape[0])
        print('val:',X_val.shape[0])
        print('test:',X_test.shape[0])

        
        model = MultilabelResNet(resnet_size,y_train.shape[1]).to(device)

        

        weights = calculate_weights(df,class_index_dict).to(device)
        loss_function = nn.BCEWithLogitsLoss(pos_weight = weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model = train(
            model = model,
            trainloader = trainloader,
            valloader = valloader,
            device = device,
            loss_function = loss_function,
            optimizer = optimizer,
            max_epochs = max_epochs,
            saving_dir = split_path,
        )


        test_metrics = evaluate(
            model = model,
            dataloader = testloader,
            loss_function = loss_function,
            device = device
        )
        print('Test')
        print_metrics(test_metrics)

        experiment = Experiment()
        experiment.add('class_index_dict',class_index_dict)
        experiment.add('resnet_size',resnet_size)

        for k,v in test_metrics.items():
            experiment.add(f'{k}_test',v)

        experiment.save(split_path)

        #save_json(test_metrics,split_path.joinpath('test_metrics.json'))



if __name__ == '__main__':
    fire.Fire(main)






        












