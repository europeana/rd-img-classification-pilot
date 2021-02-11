#to do: clean dependencies

from notebook_env import *
from ds_utils import *
import models
import torch
from PIL import Image
import time
from torch.utils.data import Dataset
import imblearn
import sklearn
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from torchvision import transforms, datasets
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from gradcam import *
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


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

        img, label = Image.open(self.X[idx]), self.y[idx]
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img,label

class InferenceDataset(Dataset):
    """
    Pytorch inference dataset class
    X: Numpy array containing the paths to the images
    """
    def __init__(self, X, transform=None):
        self.transform = transform
        self.X = X
        self.N = X.shape[0]   

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = str(self.X[idx])
        img = Image.open(img_path).convert('RGB')
                
        if self.transform:
            img = self.transform(img)

        return img,img_path



def prepare_dataset(**kwargs):
    
    X = kwargs.get('X',None)
    y = kwargs.get('y',None)
    input_size = kwargs.get('input_size',224)
    batch_size = kwargs.get('batch_size',16)
    num_workers = kwargs.get('num_workers',8)
    img_aug = kwargs.get('img_aug',None)
    splits = kwargs.get('splits',True)

    if X is None:
        raise Exception('X needs to be provided')
        
    base_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # if y is None:
    #     test_dataset = InferenceDataset(X,transform=base_transform)
    #     testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers,drop_last=False)
    #     trainloader = None
    #     return [{'trainloader':trainloader, 'testloader':testloader}]

    
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

    if not splits:
        train_dataset = TrainingDataset(X,y,transform=train_transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers,drop_last=True)
        testloader = None
        return [{'trainloader':trainloader, 'testloader':testloader}]

    
    #train-validation-test splits
    skf = StratifiedKFold(n_splits=splits)
    sk_splits = skf.split(X, y)
    splits_list = []
    for train_val_index, test_index in sk_splits:

        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]
        
        val_skf = StratifiedKFold(n_splits=splits)
        val_sk_splits = val_skf.split(X_train_val, y_train_val)
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
    



class CallBack():
    def __init__(self,**kwargs):
        self.validate = kwargs.get('validate',False)
        self.early_stopping = kwargs.get('early_stopping',False)
        self.save_best = kwargs.get('save_best',False)
        self.return_best = kwargs.get('return_best',False)
        self.path = kwargs.get('path',None)
        if self.path is not None:
            create_dir(self.path)

        self.test_loss_thres = kwargs.get('test_loss_thres',1e6)
        self.patience = kwargs.get('patience',5)
        self.stop = False
        self.save = False
        self.counter = 0
         
    def check(self,test_loss):

        if test_loss < self.test_loss_thres:
            self.test_loss_thres = test_loss
            self.counter = 0
            if self.save_best:
                self.save = True
        else:
            self.counter += 1
            self.save = False
        
        if self.early_stopping:
            if self.counter > self.patience:
                self.stop = True

        return self


def train(**kwargs):
    # todo: sample train data for metrics
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3


    #to do: add assertions
    epochs = kwargs.get('epochs',10)
    weighted_loss = kwargs.get('weighted_loss',True)
    learning_rate = kwargs.get('learning_rate',0.0001)
    model = kwargs.get('model',None)
    loss_function = kwargs.get('loss_function',None)
    optimizer = kwargs.get('optimizer',None)
    device = kwargs.get('device',None)
    encoding_dict = kwargs.get('encoding_dict',None)
    model_name = kwargs.get('model_name',None)
    trainloader = kwargs.get('trainloader',None)
    valloader = kwargs.get('valloader',None)
    callback = kwargs.get('callback',CallBack())

    best_loss = 1e6

    experiment_path = callback.path
    if callback.save_best:
        create_dir(experiment_path)  

    #initialize metrics
    loss_train_list = []
    loss_test_list = []
    acc_test_list = []
    f1_test_list = []
    precision_test_list = []
    recall_test_list = []
    specificity_test_list = []
    sensitivity_test_list = []
    recall_test_list = []
    cm_test_list = []
        
    print(f'Training for {epochs} epochs \n')
    start_train = time.time()
    # loop over epochs 
    for epoch in range(epochs):

        train_loss = 0.0
        # loop over batches
        for i, (inputs,labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(inputs.shape[0])
            loss = loss_function(model(inputs), labels.long())/inputs.shape[0]
            #backpropagate and update
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*inputs.shape[0]
                
        train_loss /= len(trainloader.dataset)
        loss_train_list.append(train_loss)
        
        if callback.validate:
            val_metrics_dict = validate_test(model,valloader,device,loss_function,encoding_dict)
            
            acc_test = val_metrics_dict['accuracy']
            f1_test = val_metrics_dict['f1']
            precision_test = val_metrics_dict['precision']
            recall_test = val_metrics_dict['recall']
            sensitivity_test = val_metrics_dict['sensitivity']
            specificity_test = val_metrics_dict['specificity']
            cm = val_metrics_dict['confusion_matrix']
            test_loss = val_metrics_dict['loss']
            
            #update test metrics
            loss_test_list.append(test_loss)
            acc_test_list.append(acc_test)
            f1_test_list.append(f1_test)
            precision_test_list.append(precision_test)
            recall_test_list.append(recall_test)
            specificity_test_list.append(specificity_test)
            sensitivity_test_list.append(sensitivity_test)
            cm_test_list.append(cm)

            print('[%d, %5d] loss: %.3f validation loss: %.3f acc: %.3f f1: %.3f precision: %.3f recall: %.3f' %
            (epoch + 1, i + 1, train_loss,test_loss,acc_test,f1_test,precision_test,recall_test))

            #callback
            callback = callback.check(test_loss)
            #save best model
            if callback.save:
                checkpoint_filename = 'checkpoint.pth'
                best_model_path = save_checkpoint(model,experiment_path,checkpoint_filename)
            if callback.stop:
                print(f'Early stopping at epoch {epoch}')
                if callback.return_best:
                    print(f'Loading best model from {best_model_path}')
                    model.load_state_dict(torch.load(best_model_path))

                break

        else:
            print('[%d, %5d] loss: %.3f ' %
            (epoch + 1, i + 1, 0))


    end_train = time.time()
    time_train = (end_train-start_train)/60.0
    print(f'\ntraining finished, it took {time_train} minutes\n')

    history = {'loss_train':loss_train_list,'loss_val':loss_test_list,
               'acc_val':acc_test_list,'confusion_matrix_val':cm_test_list,
               'f1_val':f1_test_list,'precision_val':precision_test_list,
               'recall_val':recall_test_list,'sensitivity_val':sensitivity_test_list,
               'specificity_val':specificity_test_list, }

    return model, history


def validate_test(net,testloader,device,loss_function,encoding_dict):
    """Returns metrics of predictions on test data"""

    n_labels = len(list(set(testloader.dataset.y)))

    ground_truth_list = []
    predictions_list = []
    loss = 0.0
    with torch.no_grad():
        for i,(images, labels) in enumerate(testloader):
            labels = torch.from_numpy(np.array(labels))
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            probabilities, predicted = torch.max(outputs.data, 1)
            loss += loss_function(outputs, labels.long()).item()

            ground_truth_list += list(labels.cpu())
            predictions_list += list(predicted.cpu())

    loss /= len(testloader.dataset)

        
    acc = sklearn.metrics.accuracy_score(ground_truth_list,predictions_list)
    f1 = sklearn.metrics.f1_score(ground_truth_list,predictions_list,average='macro')
    precision = sklearn.metrics.precision_score(ground_truth_list,predictions_list,average='macro')
    recall = sklearn.metrics.recall_score(ground_truth_list,predictions_list,average='macro')
    sensitivity = imblearn.metrics.sensitivity_score(ground_truth_list, predictions_list, average='macro')
    specificity = imblearn.metrics.specificity_score(ground_truth_list, predictions_list, average='macro')
    cm = sklearn.metrics.confusion_matrix(ground_truth_list,predictions_list,labels = np.arange(n_labels))

    return {
        'accuracy':acc,
        'f1':f1,
        'precision':precision, 
        'recall':recall,  
        'loss':loss,
        'sensitivity':sensitivity,
        'specificity':specificity,
        'confusion_matrix': cm, 
        'ground_truth_list':ground_truth_list,
        'predictions_list':predictions_list}

    #return metrics_dict


    

def save_XAI(model,X_test,ground_truth_list,predictions_list,split_path,device,encoding_dict):
    
    XAI_path = os.path.join(split_path,'XAI')
    create_dir(XAI_path)

    N = 20
    n_correct = 0
    n_incorrect = 0
    save = True
    for img_path,ground,pred in zip(X_test,ground_truth_list,predictions_list):

        fname = os.path.split(img_path)[1]
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


            fname = '_'.join((label,fname))

            #load img
            image = Image.open(img_path).convert('RGB')

            transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            heatmap_layer = model.net.layer4[2].conv2
            image_interpretable,_,_ = grad_cam(model, image, heatmap_layer, transform,device)


            fig,ax = plt.subplots(1,2,figsize=(20,20))
            ax[0].imshow(image)
            ax[0].axis('off')
            ax[1].imshow(image_interpretable)
            ax[1].axis('off')
            plt.savefig(os.path.join(XAI_path,fname))

        else:
            continue





    




def save_checkpoint(net,dest_path,checkpoint_filename):
    model_path = os.path.join(dest_path, checkpoint_filename)
    print(f'checkpoint saved at {model_path}')
    save_model(net,model_path)
    return model_path

def save_model(net,model_path):
    torch.save(net.state_dict(), model_path)

def predict(net,dataloader,device,encoding_dict,dest_path=None):
    img_path_list = []
    predicted_label_list = []
    confidence_list = []
    with torch.no_grad():
        for i,(images, img_path) in enumerate(dataloader):
            images = images.to(device)
            outputs = net(images)
            conf, predicted = torch.max(outputs.data, 1)

            img_path_list += list(img_path)
            predicted_label_list += list(predicted.cpu().numpy())
            confidence_list += list(conf.cpu().numpy())

    predicted_cat_list = [encoding_dict[label] for label in predicted_label_list]         
    df = pd.DataFrame(
        {'img_path':img_path_list,
        'category':predicted_cat_list,
        'label':predicted_label_list, 
        'confidence':confidence_list}
        )
    
    if dest_path:
        #save dataframe
        filename = 'predictions'+time_stamp()+'.csv'
        file_path = os.path.join(dest_path,filename)
        df.to_csv(file_path,index=False)
        print(f'Results at {file_path}')

    return df
