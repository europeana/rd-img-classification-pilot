import time
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import imblearn
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import seaborn as sn
import matplotlib.pyplot as plt

from ds_utils import *
from gradcam import *
import models

def check_args(kwargs,requ_args):
    for arg_name in requ_args:
        if not kwargs.get(arg_name):
            raise ValueError(f'{arg_name} needs to be provided')

class ImgAugTransform:
    """Class for including image augmentation in pytorch transforms"""
    def __init__(self,img_aug):
        self.aug = img_aug
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img).copy()

class TrainingDataset(Dataset):
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
        self.N = y.shape[0]   

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.X[idx]
        img, label = Image.open(img_path), self.y[idx]
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img,label,img_path


def make_train_val_test_splits(X,y,**kwargs):
    """
    Function for making train, validation and test splits for crossvalidation
    X: numpy array with the path to the images
    y: numpy array with the encoded labels
    Output: list of dictionaries with pytorch dataloaders for train, validation, test
    """

    input_size = kwargs.get('input_size',224)
    batch_size = kwargs.get('batch_size',16)
    num_workers = kwargs.get('num_workers',4)
    img_aug = kwargs.get('img_aug',None)
    splits = kwargs.get('splits',10)

  
    base_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if img_aug:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            ImgAugTransform(img_aug),
            transforms.ToTensor(),
            # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = base_transform

    #train-validation-test splits
    skf = StratifiedKFold(n_splits=splits)
    sk_splits = skf.split(X, y)
    splits_list = []
    for train_val_index, test_index in sk_splits:

        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]
        
        val_sk_splits = skf.split(X_train_val, y_train_val)
        train_index,val_index = next(val_sk_splits)
        
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        train_dataset = TrainingDataset(X_train,y_train,transform=train_transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

        val_dataset = TrainingDataset(X_val,y_val,transform=base_transform)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

        test_dataset = TrainingDataset(X_test,y_test,transform=base_transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)

        splits_list.append({'trainloader':trainloader,'valloader':valloader,'testloader':testloader})

    return splits_list
    

def train(epochs = 100, patience = 10,**kwargs):
    """    
    Trains the model
    Required arguments:
    model: 
    loss_function:
    ...
    """
    requ_args = ['model','loss_function','optimizer','trainloader','valloader','device','saving_dir','encoding_dict']
    check_args(kwargs,requ_args)

    model = kwargs.get('model')
    loss_function = kwargs.get('loss_function')
    optimizer = kwargs.get('optimizer')
    trainloader = kwargs.get('trainloader')
    valloader = kwargs.get('valloader')
    device = kwargs.get('device')
    saving_dir = kwargs.get('saving_dir')
    encoding_dict = kwargs.get('encoding_dict')

    best_loss = 1e6
    counter = 0
    experiment_path = saving_dir
    create_dir(experiment_path) 
         
    #initialize metrics
    history = {
        'loss_train':[],
        'loss_val':[],
        'accuracy_val':[],
        'confusion_matrix_val':[],
        'f1_val':[],
        'precision_val':[],
        'recall_val':[],
        'sensitivity_val':[],
        'specificity_val':[], 
               }
    
    print(f'Training for {epochs} epochs \n')
    start_train = time.time()
    # loop over epochs 
    for epoch in range(epochs):
        train_loss = 0.0
        # loop over batches
        for i, (inputs,labels,_) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels.long())/inputs.shape[0]
            #backpropagate and update
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.shape[0]

        train_loss /= len(trainloader.dataset)
        
        #evaluate model on validation data
        val_metrics_dict, _, _,_ = validate(
            model = model,
            testloader = valloader,
            device = device,
            loss_function = loss_function,
            encoding_dict = encoding_dict)

        history['loss_train'].append(train_loss)
        for k,v in val_metrics_dict.items():
            history[f'{k}_val'].append(v)


        loss_train = history['loss_train'][-1]
        loss_val = history['loss_val'][-1]
        accuracy_val = history['accuracy_val'][-1]
        f1_val = history['f1_val'][-1]
        precision_val = history['precision_val'][-1]
        recall_val = history['recall_val'][-1]
        print(f'[{epoch}] train loss: {loss_train:.3f} validation loss: {loss_val:.3f} acc: {accuracy_val:.3f} f1: {f1_val:.3f} precision: {precision_val:.3f} recall: {recall_val:.3f}')

        #save checkpoint if model improves
        if  loss_val < best_loss:
            checkpoint_path = os.path.join(experiment_path,'checkpoint.pth')
            torch.save(model.state_dict(),checkpoint_path)
            best_loss = loss_val
            counter = 0
        else:
            counter += 1

        #early stopping
        if counter > patience:
            print(f'Early stopping at epoch: {epoch}')
            break

    
    end_train = time.time()
    time_train = (end_train-start_train)/60.0
    print(f'\ntraining finished, it took {time_train} minutes\n')
    #load best model
    model.load_state_dict(torch.load(checkpoint_path))

    return model, history


def validate(**kwargs):
    """Returns metrics of predictions on test data"""

    requ_args = ['model','loss_function','testloader','device','encoding_dict']
    check_args(kwargs,requ_args)

    model = kwargs.get('model')
    loss_function = kwargs.get('loss_function')
    testloader = kwargs.get('testloader')
    device = kwargs.get('device')
    encoding_dict = kwargs.get('encoding_dict')
    

    n_labels = len(list(set(testloader.dataset.y)))

    ground_truth_list = []
    predictions_list = []
    img_path_list = []
    loss = 0.0
    with torch.no_grad():
        for i,(images, labels,img_path) in enumerate(testloader):
            labels = torch.from_numpy(np.array(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities, predicted = torch.max(outputs.data, 1)
            loss += loss_function(outputs, labels.long()).item()

            ground_truth_list += list(labels.cpu())
            predictions_list += list(predicted.cpu())
            img_path_list += list(img_path)

    loss /= len(testloader.dataset)

    metrics_dict = {
        'loss':loss,
        'accuracy':sklearn.metrics.accuracy_score(ground_truth_list,predictions_list),
        'f1':sklearn.metrics.f1_score(ground_truth_list,predictions_list,average='macro'),
        'precision':sklearn.metrics.precision_score(ground_truth_list,predictions_list,average='macro'), 
        'recall':sklearn.metrics.recall_score(ground_truth_list,predictions_list,average='macro'),  
        'sensitivity':imblearn.metrics.sensitivity_score(ground_truth_list, predictions_list, average='macro'),
        'specificity':imblearn.metrics.specificity_score(ground_truth_list, predictions_list, average='macro'),
        'confusion_matrix': sklearn.metrics.confusion_matrix(ground_truth_list,predictions_list,labels = np.arange(n_labels)), 
        }

    return metrics_dict, ground_truth_list, predictions_list, img_path_list

def save_XAI(model,test_image_list,ground_truth_list,predictions_list,split_path,device,encoding_dict):

    #to do:
    #sample a random subset of index
    #format output file [gt:,pred:]

    model.eval()
    
    XAI_path = os.path.join(split_path,'XAI')
    create_dir(XAI_path)

    N = 20
    n_correct = 0
    n_incorrect = 0
    save = True
    for img_path,ground,pred in zip(test_image_list,ground_truth_list,predictions_list):
        
        #get an equal amount of correctly classified and missclassifications
        if ground == pred:
            label = f'correct_{encoding_dict[ground.item()]}'
            n_correct += 1
            if n_correct > N:
                save = False
        else:
            label = f'true_{encoding_dict[ground.item()]}_pred_{encoding_dict[pred.item()]}'
            n_incorrect += 1
            if n_incorrect > N:
                save = False

        if save:

            transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            #load img
            image = Image.open(img_path).convert('RGB')
            #layer for the visualization
            heatmap_layer = model.net.layer4[1].conv2
            #apply gradcam
            image_interpretable,_,_ = grad_cam(model, image, heatmap_layer, transform,device)
            #save XAI
            fname = os.path.split(img_path)[1]
            fname = '_'.join((label,fname))
            fig,ax = plt.subplots(1,2,figsize=(20,20))
            ax[0].imshow(image)
            ax[0].axis('off')
            ax[1].imshow(image_interpretable)
            ax[1].axis('off')
            plt.savefig(os.path.join(XAI_path,fname))
            plt.close()

        else:
            continue


def save_model(net,model_path):
    torch.save(net.state_dict(), model_path)

#plotting
def mean_std_metric(metric):
    mean = np.mean(metric,axis=0)
    std = np.std(metric,axis=0)
    return mean, std

def plot_mean_std(metrics_dict,title,fontsize=20,y_lim = 100):
        
    fig,ax = plt.subplots(figsize = (10,10))
    for k,v in metrics_dict.items():
        mean, std = mean_std_metric(v)
        ax.plot(mean,label = k)
        ax.fill_between(range(mean.shape[0]), mean-std, mean+std, alpha = 0.5)
        ax.set_title(title,fontsize=fontsize)
        ax.set_ylim((0.0,y_lim))
        ax.set_xlabel('epochs',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        ax.grid()
        
def plot_conf_matrix(cm_array,columns,font_scale=2):

    df_cm = pd.DataFrame(cm_array, index = columns,
                      columns = columns)

    sn.set(font_scale=font_scale)  
    plt.figure(figsize = (10,10))
    sn.heatmap(df_cm, annot=True,cmap="YlGnBu",fmt="d",annot_kws={"size": 16})
    plt.xlabel('predicted')
    plt.ylabel('ground truth')
    
def read_train_history(results_path,metrics_list,max_epochs):

    metrics_dict = {k:[] for k in metrics_list}
    metrics_dict.update({f'{k}_test':[] for k in metrics_list})
    metrics_dict['loss_val'] = []
    metrics_dict['loss_train'] = []

    for split in os.listdir(results_path):
        training_info_path = os.path.join(results_path,split,'training_info.pth')
        training_info = torch.load(training_info_path)
        encoding_dict = training_info['encoding_dict']

        for metric in ['loss_val','loss_train']:
            loss_arr = np.array(training_info[metric])
            loss_arr = np.pad(loss_arr,(0,max_epochs-loss_arr.shape[0]),'edge')
            metrics_dict[metric].append(list(loss_arr))  

        for metric in metrics_list:
            metric_arr = np.array(training_info[f'{metric}_val'])
            metric_arr = np.pad(metric_arr,(0,max_epochs-metric_arr.shape[0]),'edge')
            metrics_dict[metric].append(list(metric_arr))

            test_metric = training_info[f'{metric}_test']
            metrics_dict[f'{metric}_test'].append(test_metric)
            
    return metrics_dict
