import time
import os
import pandas as pd
import numpy as np
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import imblearn
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import seaborn as sn
import matplotlib.pyplot as plt

from ds_utils import *
from gradcam import *

from imgaug import augmenters as iaa



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

class ResNet(nn.Module):
    def __init__(self,size, output_size):
        super(ResNet, self).__init__()

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
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.net(x)
        out = self.sm(out)
        return out

def get_class_weights(y_encoded,encoding_dict):
    """Calculates the weights for the Cross Entropy loss """
    data_dict = get_imgs_per_cat(y_encoded)       
    N = sum(data_dict.values())
    #calculate weights as the inverse of the frequency of each class
    weights = []
    for k in data_dict.keys(): 
        weights.append(N/data_dict[k])
    return weights

def get_imgs_per_cat(y_encoded):
    #count the images in each category
    data_dict = {}
    for el in y_encoded:
        if el not in data_dict.keys():
            data_dict.update({el:1})
        else:
            data_dict[el] += 1
    return data_dict

def label_encoding(y):
    le = preprocessing.LabelEncoder()
    y_encoded = le.fit_transform(y)
    encoding_dict = {}
    for cat in le.classes_:
        label = le.transform(np.array([cat]))[0]
        encoding_dict.update({int(label):cat}) 
    return y_encoded, encoding_dict

class Experiment():
    def __init__(self):
        self.info = {}

    def add(self,key,value):
        self.info.update({key:value})
        return self

    def show(self):
        print(f'keys: {self.info.keys()}\n')
        for k,v in self.info.items():
            print(f'{k}: {v}\n')

    def save(self,dest_path):
        filename = 'training_info.pth'
        info_file_path = os.path.join(dest_path,filename)
        torch.save(self.info, info_file_path)


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


def save_class_index(class_index,path):
    import json
    with open(os.path.join(path,'class_index.json'),'w') as f:
        json.dump(class_index,f)
    

def train(epochs = 100, patience = 10,**kwargs):
    """    
    Trains the model
    Required arguments:
    model: 
    loss_function:
    ...
    """
    requ_args = ['model','loss_function','optimizer','trainloader','valloader','device','saving_dir']
    check_args(kwargs,requ_args)

    model = kwargs.get('model')
    loss_function = kwargs.get('loss_function')
    optimizer = kwargs.get('optimizer')
    trainloader = kwargs.get('trainloader')
    valloader = kwargs.get('valloader')
    device = kwargs.get('device')
    saving_dir = kwargs.get('saving_dir')
    #encoding_dict = kwargs.get('encoding_dict')

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
        val_metrics_dict, _, _,_ = evaluate(
            model = model,
            dataloader = valloader,
            device = device,
            loss_function = loss_function)

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

    # to do: save class_index dict
    #load best model
    model.load_state_dict(torch.load(checkpoint_path))

    return model, history


def evaluate(**kwargs):
    """Returns metrics of predictions on test data"""

    requ_args = ['model','loss_function','dataloader','device']
    check_args(kwargs,requ_args)

    model = kwargs.get('model')
    loss_function = kwargs.get('loss_function')
    dataloader = kwargs.get('dataloader')
    device = kwargs.get('device')
    #encoding_dict = kwargs.get('class_index_dict')
    

    n_labels = len(list(set(dataloader.dataset.y)))

    ground_truth_list = []
    predictions_list = []
    img_path_list = []
    loss = 0.0
    with torch.no_grad():
        for i,(images, labels,img_path) in enumerate(dataloader):
            labels = torch.from_numpy(np.array(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities, predicted = torch.max(outputs.data, 1)
            loss += loss_function(outputs, labels.long()).item()

            ground_truth_list += list(labels.cpu())
            predictions_list += list(predicted.cpu())
            img_path_list += list(img_path)

    loss /= len(dataloader.dataset)

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

def save_XAI(N = 20, **kwargs):
    """
    Comment
    """

    #to do: change name of function

    #to do: return list of path to original images, gt, pred, and XAI images
    
    
    requ_args = ['model','test_images_list','ground_truth_list','predictions_list','saving_dir','device','class_index_dict']
    check_args(kwargs,requ_args)
    
    model = kwargs.get('model')
    test_images_list = kwargs.get('test_images_list')
    ground_truth_list = kwargs.get('ground_truth_list')
    predictions_list = kwargs.get('predictions_list')
    saving_dir = kwargs.get('saving_dir')
    device = kwargs.get('device')
    class_index_dict = kwargs.get('class_index_dict')

    
    model.eval()
    
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    XAI_path = os.path.join(saving_dir,'XAI')
    create_dir(XAI_path)
    
    #sample of test images
    rand_idx = np.random.randint(len(test_images_list),size = N)
    sample_img_path_list = [test_images_list[idx] for idx in rand_idx]
    sample_gt_list = [ground_truth_list[idx] for idx in rand_idx]
    sample_pred_list = [predictions_list[idx] for idx in rand_idx]
    
    for img_path,label,pred in zip(sample_img_path_list,sample_gt_list,sample_pred_list):
        #load img
        image = Image.open(img_path).convert('RGB')
        #layer for the visualization
        heatmap_layer = model.net.layer4[1].conv2
        #apply gradcam
        image_interpretable,_,_ = grad_cam(model, image, heatmap_layer, transform,device)
        #save XAI
        fname = os.path.split(img_path)[1]
        
        ground_truth = class_index_dict[label.item()]
        prediction = class_index_dict[pred.item()]
        
        fig,ax = plt.subplots(1,2,figsize=(20,20))
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].title.set_text(f'Ground truth: {ground_truth}')
        ax[1].imshow(image_interpretable)
        ax[1].axis('off')
        ax[1].title.set_text(f'Prediction: {prediction}')
        
        plt.savefig(os.path.join(XAI_path,'_'.join((f'[gt:{ground_truth},pred:{prediction}]',fname))))
        plt.show()

    model.train()



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

    img_aug = kwargs.get('img_aug',0.5)

    if not img_aug:
        img_aug_seq = None
    else:
        sometimes = lambda augmentation: iaa.Sometimes(img_aug, augmentation)
        img_aug_seq = iaa.Sequential([
            iaa.Fliplr(img_aug),
            sometimes(iaa.ChangeColorTemperature((1100, 10000))),

            sometimes(iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 2.0)),
                iaa.AddToHueAndSaturation((-10, 10))

            ]))

        ])


    create_dir(saving_dir)
    experiment_path = os.path.join(saving_dir,experiment_name)
    create_dir(experiment_path)

    #load data
    df = path2DataFrame(data_dir)
    
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
        img_aug = img_aug_seq,
        num_workers = num_workers,
        batch_size = batch_size,
        splits = 10,
    )

    #crossvalidation
    for i,split in enumerate(data_splits):
        print(f'split {i}\n')
        split_path = os.path.join(experiment_path,f'split_{i}')
        create_dir(split_path)
        
        trainloader = split['trainloader']
        valloader = split['valloader']
        testloader = split['testloader']

        print('size train: {}'.format(len(trainloader.dataset)))
        print('size val: {}'.format(len(valloader.dataset)))
        print('size test: {}'.format(len(testloader.dataset)))

        save_class_index(class_index_dict,split_path)
        
        #initialize model
        model = ResNet(resnet_size,n_classes).to(device)
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
        encoding_dict = training_info['class_index_dict']

        for metric in ['loss_val','loss_train']:
            loss_arr = np.array(training_info[metric])
            loss_arr = np.pad(loss_arr,(0,max_epochs-loss_arr.shape[0]),'edge')
            metrics_dict[metric].append(list(loss_arr))  

        for metric in metrics_list:
            metric_arr = np.array(training_info[f'{metric}_val'])
            metric_arr = np.pad(metric_arr,(0,max_epochs-metric_arr.shape[0]),'edge')
            metrics_dict[metric].append(list(metric_arr))

            #test_metric = training_info[f'{metric}_test']
            #metrics_dict[f'{metric}_test'].append(test_metric)
            
    return metrics_dict
