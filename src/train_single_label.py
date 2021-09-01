import argparse
import os
import fire
from torch_utils import *

def main(**kwargs):
  data_dir = kwargs.get('data_dir')
  saving_dir = kwargs.get('saving_dir')
  learning_rate = kwargs.get('learning_rate',1e-4)
  epochs = kwargs.get('epochs',100)
  patience = kwargs.get('patience',10)
  resnet_size = kwargs.get('resnet_size',18)
  num_workers = kwargs.get('num_workers',8)
  batch_size = kwargs.get('batch_size',16)
  weighted_loss = kwargs.get('weighted_loss',True)
  input_size = kwargs.get('input_size',64)
  sample = kwargs.get('sample',1.0)
  crossvalidation = kwargs.get('crossvalidation',False)

  train_single_label(
      data_dir = data_dir,
      input_size = input_size,
      saving_dir = saving_dir,
      learning_rate = learning_rate,
      epochs = epochs,
      patience = patience,
      resnet_size = resnet_size,
      num_workers = num_workers,
      batch_size = batch_size,
      weighted_loss = weighted_loss,
      sample = sample,
      crossvalidation = crossvalidation
  )
  return


if __name__ == '__main__':
    fire.Fire(main)

    """
    Script for training crossvalidation

    Usage:

      python src/train.py --data_dir training_data --epochs 100 --patience 10 --experiment_name model_training --img_aug 0.5

    Parameters:

      data_dir: directory containing the dataset organized in 
                subdirectories for the different categories
                Required

      saving_dir: directory for saving the training results. 
                  If not specified this will be the root path of the repository

      experiment_name: tag for the results
                      Default: results_training

      learning_rate: Default: 0.00001

      epochs: Default: 100

      patience: Number of epochs for early stopping
                Default: 10

      resnet_size: size of the model. The allowed sizes are 18,34,50,101,152
                   Default: 34

      num_workers: Default: 4

      batch_size: Default: 64

      weighted_loss: Whether to penalize missclassifications for the minority classes
                     Recommended for imbalanced datasets.
                     Default: True

      img_aug: Whether to use image augmentation
                Default: True

    """

    






        












