import argparse
import os
from torch_utils import *


#import pandas as pd
#import torch
#from ds_utils import *





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
    parser.add_argument('--img_aug', required=False)


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
      epochs = 100
    else:
      epochs = int(args.epochs)

    if not args.resnet_size:
      resnet_size = 34
    else:
      resnet_size = int(args.resnet_size)

    if not args.patience:
      patience = 10
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

    if not args.img_aug:
      img_aug = None
    else:
      img_aug = float(args.img_aug)


    train_crossvalidation(
        data_dir = args.data_dir ,
        saving_dir = saving_dir,
        experiment_name = experiment_name,
        learning_rate = learning_rate,
        epochs = epochs,
        patience = patience,
        resnet_size = resnet_size,
        num_workers = num_workers,
        batch_size = batch_size,
        weighted_loss = weighted_loss,
        img_aug = img_aug
    )




        












