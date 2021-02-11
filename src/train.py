from notebook_env import *

import torch
import os
import pandas as pd

from ds_utils import *
from torch_utils import *
from imgaug import augmenters as iaa

if __name__ == '__main__':

    #to do: change encoding dict by class_index_dict

    experiment_name = 'testing'
    learning_rate = 0.00001
    epochs = 5
    resnet_size = 34 # allowed sizes: 18,34,50,101,152
    num_workers = 4
    batch_size = 64
    weighted_loss = True
    img_aug = None

    patience = 1

    device = torch.device('cuda:0')


    results_path = '../results'
    create_dir(results_path)
    experiment_path = os.path.join(results_path,experiment_name)
    create_dir(experiment_path)

    #to do: load from a unique directory
    data_path = '/home/jcejudo/rd-img-classification-pilot/training_data/ec'
    ec_df = path2DataFrame(data_path)

    data_path = '/home/jcejudo/rd-img-classification-pilot/training_data/getty'
    getty_df = path2DataFrame(data_path)

    df = pd.concat((ec_df,getty_df))
    
    #remove after testing
    df = df.sample(frac=0.1)

    X = df['file_path'].values
    y = df['category'].values
    y_encoded, encoding_dict = label_encoding(y)
    n_classes = len(encoding_dict)

    #set loss
    if weighted_loss:
        weights = get_class_weights(y_encoded,encoding_dict)
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
            model,
            loss_function,
            optimizer,
            trainloader,
            valloader,
            device,
            split_path,
            encoding_dict,
            epochs = 10,
            patience = 1)

        # evaluate on test data
        metrics_dict, ground_truth_list, predictions_list = validate(model,testloader,device,loss_function,encoding_dict)
        #generate heatmaps using GradCAM for some test images
        test_image_list = testloader.dataset.X
        save_XAI(model,test_image_list,ground_truth_list,predictions_list,split_path,device,encoding_dict)

        #print test metrics
        for k,v in metrics_dict.items():
            print(f'{k}_test: {v}')

        
        experiment = Experiment()
        #experiment.add('learning_rate',learning_rate)
        #experiment.add('optimizer',optimizer)
        #experiment.add('loss_function',loss_function)

        #experiment.add('batch_size',batch_size)
        #experiment.add('num_workers',num_workers)

        #experiment.add('epochs',epochs)
        #experiment.add('weights',weights)

        experiment.add('encoding_dict',encoding_dict)
        experiment.add('model',model)
        experiment.add('resnet_size',resnet_size)

        for k,v in metrics_dict.items():
            experiment.add(f'{k}_test',v)

        for k,v in history.items():
            experiment.add(k,v)

        experiment.save(split_path)
        












