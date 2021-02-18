import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from ds_utils import *
from torch_utils import *
from imgaug import augmenters as iaa


def check_args(kwargs,requ_args):
    for arg_name in requ_args:
        if not kwargs.get(arg_name):
            raise ValueError(f'{arg_name} needs to be provided')


def train_crossvalidation(**kwargs):

    check_args(kwargs,['data_dir','saving_dir'])

    data_dir = kwargs.get('data_dir',None)
    saving_dir = kwargs.get('saving_dir',None)

    experiment_name = kwargs.get('experiment_name','model_training')
    learning_rate = kwargs.get('learning_rate',0.00001)
    epochs = kwargs.get('epochs',100)
    patience = kwargs.get('patience',10)
    resnet_size = kwargs.get('resnet_size',34) # allowed sizes: 18,34,50,101,152
    num_workers = kwargs.get('num_workers',4)
    batch_size = kwargs.get('batch_size',64)
    weighted_loss = kwargs.get('weighted_loss',True)

    #to do: include image augmentation    
    
    # prob_aug = 0.5
    # sometimes = lambda augmentation: iaa.Sometimes(prob_aug, augmentation)
    # img_aug = iaa.Sequential([
    #     iaa.Fliplr(prob_aug),
    #     sometimes(iaa.Crop(percent=(0, 0.2))),
    #     sometimes(iaa.ChangeColorTemperature((1100, 10000))),

    #     sometimes(iaa.OneOf([
    #         iaa.GaussianBlur(sigma=(0, 2.0)),
    #         iaa.AddToHueAndSaturation((-10, 10))

    #     ]))

    # ])

    img_aug = None


    results_path = os.path.join(ROOT_DIR,'results')
    create_dir(results_path)
    experiment_path = os.path.join(results_path,experiment_name)
    create_dir(experiment_path)

    #load data
    data_path = os.path.join(ROOT_DIR,'new_training')
    df = path2DataFrame(data_path)
    
    #remove after testing
    df = df.sample(frac=0.1)

    X = df['file_path'].values
    y = df['category'].values
    y_encoded, class_index_dict = label_encoding(y)
    n_classes = len(class_index_dict)

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #set loss
    if weighted_loss:
        weights = get_class_weights(y_encoded,class_index_dict)
        loss_function = nn.CrossEntropyLoss(reduction ='sum',weight=torch.FloatTensor(weights).to(device))           
    else:
        loss_function = nn.CrossEntropyLoss(reduction='sum')


    data_splits = make_train_val_test_splits(
        X,
        y_encoded,
        img_aug = img_aug,
        num_workers = num_workers,
        batch_size = batch_size,
        splits = 10,
    )

    #crossvalidation
    for i,split in enumerate(data_splits):
        print(f'split {i}\n')
        split_path = os.path.join(experiment_path,f'split_{i}')
        
        trainloader = split['trainloader']
        valloader = split['valloader']
        testloader = split['testloader']

        print('size train: {}'.format(len(trainloader.dataset)))
        print('size val: {}'.format(len(valloader.dataset)))
        print('size test: {}'.format(len(testloader.dataset)))
        
        #initialize model
        model = models.ResNet(resnet_size,n_classes).to(device)
        #set optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model, history = train(
            model = model,
            loss_function = loss_function,
            optimizer = optimizer,
            trainloader = trainloader,
            valloader = valloader,
            device = device,
            saving_dir = split_path,
            epochs = epochs,
            patience = patience)

        # evaluate on test data
        metrics_dict, ground_truth_list, predictions_list, test_images_list = evaluate(
            model = model,
            dataloader = testloader,
            device = device,
            loss_function = loss_function
            )

        #generate heatmaps using GradCAM for some test images
        save_XAI(
            model = model,
            test_images_list = test_images_list,
            ground_truth_list = ground_truth_list,
            predictions_list = predictions_list,
            saving_dir = split_path,
            device = device,
            class_index_dict = class_index_dict)

        #print test metrics
        for k,v in metrics_dict.items():
            print(f'{k}_test: {v}')

        #save training history
        experiment = Experiment()
        experiment.add('class_index_dict',class_index_dict)
        experiment.add('model',model)
        experiment.add('resnet_size',resnet_size)

        for k,v in metrics_dict.items():
            experiment.add(f'{k}_test',v)

        for k,v in history.items():
            experiment.add(k,v)

        experiment.save(split_path)

    return 


if __name__ == '__main__':

    ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--saving_dir', required=False)

    parser.add_argument('--experiment_name', required=False)
    parser.add_argument('--learning_rate', required=False)
    parser.add_argument('--epochs', required=False)

    parser.add_argument('--patience', required=False)
    parser.add_argument('--resnet_size', required=False)
    parser.add_argument('--num_workers', required=False)
    parser.add_argument('--batch_size', required=False)
    parser.add_argument('--weighted_loss', required=False)


    args = parser.parse_args()

    if not args.experiment_name:
      experiment_name = 'model_training'
    else:
      experiment_name = args.experiment_name

    if not args.saving_dir:
      saving_dir = ROOT_DIR
    else:
      saving_dir = args.saving_dir

    if not args.learning_rate:
      learning_rate = 0.00001
    else:
      learning_rate = float(args.learning_rate)

    if not args.epochs:
      epochs = 2
    else:
      epochs = int(args.epochs)

    if not args.resnet_size:
      resnet_size = 34
    else:
      resnet_size = int(args.resnet_size)

    if not args.patience:
      patience = 1
    else:
      patience = int(args.patience)

    if not args.num_workers:
      num_workers = 4
    else:
      num_workers = int(args.num_workers)

    if not args.batch_size:
      batch_size = 64
    else:
      batch_size = int(args.batch_size)

    if not args.weighted_loss:
      weighted_loss = True
    else:
      weighted_loss = bool(args.weighted_loss)


    train_crossvalidation(
        data_dir = args.data_dir ,
        saving_dir = args.saving_dir,
        experiment_name = experiment_name,
        learning_rate = learning_rate,
        epochs = epochs,
        patience = patience,
        resnet_size = resnet_size,
        num_workers = num_workers,
        batch_size = batch_size,
        weighted_loss = weighted_loss
    )




        












