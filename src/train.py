import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ds_utils import *
from torch_utils import *
from imgaug import augmenters as iaa

if __name__ == '__main__':

    #to do: change encoding dict by class_index_dict

    #to do: add command line interface?

    ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    experiment_name = 'testing'
    learning_rate = 0.00001
    epochs = 2
    patience = 1
    resnet_size = 34 # allowed sizes: 18,34,50,101,152
    num_workers = 4
    batch_size = 64
    weighted_loss = True
    
    
    prob_aug = 0.5
    sometimes = lambda augmentation: iaa.Sometimes(prob_aug, augmentation)
    img_aug = iaa.Sequential([
        iaa.Fliplr(prob_aug),
        sometimes(iaa.Crop(percent=(0, 0.2))),
        sometimes(iaa.ChangeColorTemperature((1100, 10000))),

        sometimes(iaa.OneOf([
            iaa.GaussianBlur(sigma=(0, 2.0)),
            iaa.AddToHueAndSaturation((-10, 10))

        ]))

    ])

    #img_aug = None


    results_path = os.path.join(ROOT_DIR,'results')
    create_dir(results_path)
    experiment_path = os.path.join(results_path,experiment_name)
    create_dir(experiment_path)

    #load data
    data_path = os.path.join(ROOT_DIR,'new_training')
    df = path2DataFrame(data_path)
    
    #remove after testing
    #df = df.sample(frac=0.1)

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
            encoding_dict = class_index_dict,
            epochs = epochs,
            patience = patience)

        # evaluate on test data
        metrics_dict, ground_truth_list, predictions_list, test_images_list = validate(
            model = model,
            testloader = testloader,
            device = device,
            loss_function = loss_function,
            encoding_dict = class_index_dict
            )


        #generate heatmaps using GradCAM for some test images
        #test_image_list = testloader.dataset.X
        #save_XAI(model,test_images_list,ground_truth_list,predictions_list,split_path,device,class_index_dict)

        save_XAI(
            model = model,
            test_images_list = test_images_list,
            ground_truth_list = ground_truth_list,
            predictions_list = predictions_list,
            split_path = split_path,
            device = device,
            encoding_dict = class_index_dict)

        #print test metrics
        for k,v in metrics_dict.items():
            print(f'{k}_test: {v}')

        #save training history
        experiment = Experiment()
        experiment.add('encoding_dict',class_index_dict)
        experiment.add('model',model)
        experiment.add('resnet_size',resnet_size)

        for k,v in metrics_dict.items():
            experiment.add(f'{k}_test',v)

        for k,v in history.items():
            experiment.add(k,v)

        experiment.save(split_path)
        












