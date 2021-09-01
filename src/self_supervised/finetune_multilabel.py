from pathlib import Path
import torch
import torch.nn as nn
import os
import lightly
import json
import fire

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

import sys

sys.path.append(ROOT_DIR)

from train_multilabel import *

from ss_models import *

class FineTuneModel(nn.Module):
    def __init__(self, model,num_ftrs,output_dim):
        super().__init__()
        
        self.net = model
        #self.fc1 = nn.Linear(num_ftrs, 256)
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_ftrs, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y_hat = self.net.backbone(x).squeeze()
        #y_hat = self.fc1(y_hat)
        #y_hat = self.relu(y_hat)
        y_hat = self.fc2(y_hat)
        y_hat = self.sigmoid(y_hat)
        return y_hat
            
def load_simcrl(simclr_results_path,n_categories,device,model_size = 18):
    
    #load config
    conf_path = simclr_results_path.joinpath('conf.json')
    with open(conf_path,'r') as f:
        conf = json.load(f)

    #load model
    model_path = simclr_results_path.joinpath('checkpoint.pth')

    num_ftrs = conf['num_ftrs']
    
    model_name = conf['model_name']

    resnet = lightly.models.ResNetGenerator('resnet-'+str(model_size))
    last_conv_channels = list(resnet.children())[-1].in_features
    backbone = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1)
    )

    if model_name == 'simclr':
        model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)
    elif model_name == 'moco':
        model = lightly.models.MoCo(backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)
        

    encoder = lightly.embedding.SelfSupervisedEmbedding(
        model,
        None,
        None,
        None
    )

    encoder.model.load_state_dict(torch.load(model_path))
    teacher = FineTuneModel(encoder.model,num_ftrs,n_categories).to(device)
    return teacher


def save_metrics(metrics,path):
  with open(path,'w') as f:
    json.dump(metrics,f)

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
  pretrained_dir = kwargs.get('pretrained_dir')
  data_dir = kwargs.get('data_dir')
  saving_dir = kwargs.get('saving_dir')
  input_size = kwargs.get('input_size',64)
  batch_size = kwargs.get('batch_size',32)
  num_workers = kwargs.get('num_workers',8)
  learning_rate = kwargs.get('learning_rate',1e-5)
  hf_prob = kwargs.get('hf_prob',0.0)
  vf_prob = kwargs.get('vf_prob',0.0)
  cj_prob = kwargs.get('cj_prob',0.0)
  gb_prob = kwargs.get('gb_prob',0.0)

  data_dir = Path(data_dir)
  df_path = Path(annotations)
  pretrained_dir = Path(pretrained_dir)
  saving_dir = Path(saving_dir)
  saving_dir.mkdir(parents=True, exist_ok=True)

  df = pd.read_csv(df_path)
  #filter images in df contained in data_path
  imgs_list = list(data_dir.iterdir())
  df['filepath'] = df['ID'].apply(lambda x:data_dir.joinpath(id_to_filename(x)+'.jpg'))
  df = df.loc[df['filepath'].apply(lambda x: Path(x) in imgs_list)]
  df['n_labels'] = df['category'].apply(lambda x: len(x.split()))
  df = df.sort_values(by='n_labels',ascending=False)
  df = df.drop_duplicates()
  print(df.shape)

  mlb = sklearn.preprocessing.MultiLabelBinarizer()

  imgs = np.array([str(path) for path in df['filepath'].values])

  labels = [item.split() for item in df['category'].values]
  labels = mlb.fit_transform(labels)

  class_index_dict = {i:c for i,c in enumerate(mlb.classes_)}

  #to do: crossvalidation

  imgs_train,imgs_evaluation,labels_train,labels_evaluation = train_test_split(imgs,labels,test_size = 0.3)
  imgs_val,imgs_test,labels_val,labels_test = train_test_split(imgs_evaluation,labels_evaluation,test_size = 0.5)

  color_jitter = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)

  kernel_size = 3
  gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

  train_transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((input_size,input_size)),
      torchvision.transforms.RandomApply([color_jitter], p=cj_prob),
      torchvision.transforms.RandomApply([gaussian_blur], p=gb_prob),
      torchvision.transforms.RandomHorizontalFlip(p=hf_prob),
      torchvision.transforms.RandomVerticalFlip(p=vf_prob),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
          mean=lightly.data.collate.imagenet_normalize['mean'],
          std=lightly.data.collate.imagenet_normalize['std'],
      )
  ])

  test_transform = transforms.Compose([
      transforms.Resize((input_size, input_size)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=lightly.data.collate.imagenet_normalize['mean'],
          std=lightly.data.collate.imagenet_normalize['std'],
      )
  ])

  # to do: crossvalidation

  trainset = MultilabelDataset(imgs_train,labels_train,transform = train_transform)
  trainloader = DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True)

  valset = MultilabelDataset(imgs_val,labels_val,transform = test_transform)
  valloader = DataLoader(
    valset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True)

  testset = MultilabelDataset(imgs_test,labels_test,transform = test_transform)
  testloader = DataLoader(
    testset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=num_workers,
    drop_last=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  n_categories = labels.shape[1]

  config_path = pretrained_dir.joinpath('conf.json')
  model_path = pretrained_dir.joinpath('checkpoint.pth')

  with open(config_path,'r') as f:
    config = json.load(f)

  ss_model = config['ss_model']
  model_size = config['model_size']
  num_ftrs = config['num_ftrs']
  benchmarking = config['benchmarking']

  resnet_dict = {
      18: torchvision.models.resnet18(pretrained=False),
      34: torchvision.models.resnet34(pretrained=False),
      50: torchvision.models.resnet50(pretrained=False),
      101: torchvision.models.resnet101(pretrained=False),
  }

  conf = {
      'resnet_size':model_size,
      'input_size':input_size,
      'ss_model':ss_model,
      'benchmarking':benchmarking,
      'num_ftrs':num_ftrs,
      'n_categories':n_categories

  }

  with open(saving_dir.joinpath('conf.json'),'w') as f:
    json.dump(conf,f)

  with open(saving_dir.joinpath('class_index.json'),'w') as f:
    json.dump(class_index_dict,f)

  backbone = resnet_dict[model_size]

  if ss_model == 'byol':

      if config['benchmarking']:
        model = BYOLModel_benchmarking(backbone,None,1,num_ftrs=num_ftrs,)

      else:
        model = BYOLModel(
            backbone,
            num_ftrs = num_ftrs,
            )

  elif ss_model == 'moco':
      if config['benchmarking']:
        model = MoCoModel_benchmarking(backbone,None,1,num_ftrs=num_ftrs,)

      else:
        model = MoCoModel(
            backbone,
            num_ftrs = num_ftrs,
            )

  model.backbone.load_state_dict(torch.load(model_path))

  model = FineTuneModel(
    model,
    num_ftrs,
    n_categories
  )

  model = model.to(device)

  weights = calculate_weights(df,class_index_dict).to(device)
  loss_function = nn.BCEWithLogitsLoss(pos_weight = weights)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1.0)
  model = train(
      model = model,
      trainloader = trainloader,
      valloader = valloader,
      device = device,
      loss_function = loss_function,
      optimizer = optimizer,
      scheduler = scheduler,
      max_epochs = max_epochs,
      saving_dir = saving_dir,
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
  #experiment.add('resnet_size',resnet_size)

  for k,v in test_metrics.items():
      experiment.add(f'{k}_test',v)

  experiment.save(saving_dir)

  return 




if __name__ == '__main__':
  fire.Fire(main)
