from notebook_env import *
from ds_utils import *
from data_wrangling import *
import models
import torch
from PIL import Image
import time
from torch.utils.data import Dataset
import imblearn
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from torchvision import transforms, datasets
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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

        img_path = self.X[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.y[idx]
                
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

def check_cuda():
    multi_gpu = False
    number_devices = torch.cuda.device_count()
    if number_devices > 1:
        multi_gpu = True
        #to do: for more than two GPUs
        device = torch.device('cuda:0,1')
        print('Using {} devices \n'.format(number_devices))
    else:
        device = torch.device('cuda:0')
        print('Using a single device \n')
    return device,multi_gpu

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

    if y is None:
        test_dataset = InferenceDataset(X,transform=base_transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=num_workers,drop_last=False)
        trainloader = None
        return [{'trainloader':trainloader, 'testloader':testloader}]

    
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
    
        
def build_model(model_name, device, multi_gpu, output_size=None, show=False):

    print(f'Building model {model_name}')
    if model_name == "CNN":
        net = models.ConvNet(output_size)

    elif model_name.startswith("resnet"):
        size = int(model_name.replace("resnet",""))
        net = models.ResNet(size,output_size)

    #multiprocessing model
    if multi_gpu:
        net = nn.DataParallel(net)
    net = net.to(device)
    
    if show:
        print(net)

    return net

def load_model(model_name,model_path,device,multi_gpu,output_size = None):
    net = build_model(model_name, device, multi_gpu, output_size=output_size, show=False)
    print(f'Loading model from {model_path}')
    net.load_state_dict(torch.load(model_path))
    print(f'Finished loading model {model_name}')
    return net

def get_best_model_path(training_results_path,monitor = 'val_loss',mode = 'min'):
    """
    monitor must be in ['train_loss','val_loss', 'val_acc',  'val_f1','val_precision', 'val_recall' ]
    mode must be in ['min', 'max']
    """
    import regex as re
    index = None
    if mode == 'min':
        ref_metric = 1e6
    else:
        ref_metric = -1e6
    model_path_list = os.listdir(training_results_path)
    model_path_list = [model_path for model_path in model_path_list if model_path.startswith('best')]
    for i,_model_path in enumerate(model_path_list):
        metrics = [float(n) for n in re.findall('\d+\.\d+',_model_path)]
        #print(metrics)
        metrics_dict = {'train_loss':metrics[0], 'val_loss':metrics[1],'val_acc':metrics[2],
                        'val_f1':metrics[3],'val_precision':metrics[4],'val_recall':metrics[5]}

        m = metrics_dict[monitor]
        if mode == 'min':
            if m < ref_metric:
                ref_metric = m
                model_path = _model_path
        else:
            if m > ref_metric:
                ref_metric = m
                model_path = _model_path
                
    return os.path.join(training_results_path,model_path)

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
    net = kwargs.get('net',None)
    loss_function = kwargs.get('loss_function',None)
    optimizer = kwargs.get('optimizer',None)
    device = kwargs.get('device',None)
    multi_gpu = kwargs.get('multi_gpu',None)
    encoding_dict = kwargs.get('encoding_dict',None)
    model_name = kwargs.get('model_name',None)
    trainloader = kwargs.get('trainloader',None)
    valloader = kwargs.get('valloader',None)
    callback = kwargs.get('callback',CallBack())

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
    auc_test_list = []
    ppv_test_list = []
    npv_test_list = []
        
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
            loss = loss_function(net(inputs), labels.long())/inputs.shape[0]
            #backpropagate and update
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*inputs.shape[0]
                
        train_loss /= len(trainloader.dataset)
        loss_train_list.append(train_loss)
        
        if callback.validate:
            val_metrics_dict = validate_test(net,valloader,device,loss_function,encoding_dict)
            
            acc_test = val_metrics_dict['accuracy']
            f1_test = val_metrics_dict['f1']
            precision_test = val_metrics_dict['precision']
            recall_test = val_metrics_dict['recall']
            sensitivity_test = val_metrics_dict['sensitivity']
            specificity_test = val_metrics_dict['specificity']
            cm = val_metrics_dict['confusion_matrix']
            test_loss = val_metrics_dict['loss']
            auc_test = val_metrics_dict['auc']
            ppv_test = val_metrics_dict['ppv']
            npv_test = val_metrics_dict['npv']
            ROC_fig = val_metrics_dict['ROC_fig']
            
            #update test metrics
            loss_test_list.append(test_loss)
            acc_test_list.append(acc_test)
            f1_test_list.append(f1_test)
            precision_test_list.append(precision_test)
            recall_test_list.append(recall_test)
            specificity_test_list.append(specificity_test)
            sensitivity_test_list.append(sensitivity_test)
            cm_test_list.append(cm)
            auc_test_list.append(auc_test)
            ppv_test_list.append(ppv_test)
            npv_test_list.append(npv_test)


            print('[%d, %5d] loss: %.3f validation loss: %.3f acc: %.3f f1: %.3f precision: %.3f recall: %.3f' %
            (epoch + 1, i + 1, train_loss,test_loss,acc_test,f1_test,precision_test,recall_test))

            #callback
            callback = callback.check(test_loss)
            #save best model
            # to do: include minimal training info in the checkpoint
            if callback.save:
                #checkpoint_filename = f'best_model_epoch_{epoch}_loss_train_{train_loss:.3f}_val_loss_{test_loss:.3f}_acc_{acc_test:.3f}_f1_{f1_test:.3f}_precision_{precision_test:.3f}_recall_{recall_test:.3f}.pth'
                checkpoint_filename = 'checkpoint.pth'
                best_model_path = save_checkpoint(net,experiment_path,checkpoint_filename)
            if callback.stop:
                print(f'Early stopping at epoch {epoch}')
                if callback.return_best:
                    print(f'Loading best model from {best_model_path}')
                    net = load_model(model_name,best_model_path,device,multi_gpu,output_size = len(encoding_dict))
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
               'specificity_val':specificity_test_list,'auc_val':auc_test_list, 
               'ppv_val':ppv_test_list,'npv_val':npv_test_list,  }

    return net, history


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
    if n_labels > 2:
        f1 = sklearn.metrics.f1_score(ground_truth_list,predictions_list,average='macro')
        precision = sklearn.metrics.precision_score(ground_truth_list,predictions_list,average='macro')
        recall = sklearn.metrics.recall_score(ground_truth_list,predictions_list,average='macro')
        sensitivity = imblearn.metrics.sensitivity_score(ground_truth_list, predictions_list, average='macro')
        specificity = imblearn.metrics.specificity_score(ground_truth_list, predictions_list, average='macro')
        cm = sklearn.metrics.confusion_matrix(ground_truth_list,predictions_list,labels = np.arange(n_labels))
        #ppv = None
        #npv = None
    else:
        pass
        # f1 = sklearn.metrics.f1_score(ground_truth_list,predictions_list)
        # precision = sklearn.metrics.precision_score(ground_truth_list,predictions_list)
        # recall = sklearn.metrics.recall_score(ground_truth_list,predictions_list)
        # sensitivity = imblearn.metrics.sensitivity_score(ground_truth_list, predictions_list)
        # specificity = imblearn.metrics.specificity_score(ground_truth_list, predictions_list)
        # cm = sklearn.metrics.confusion_matrix(ground_truth_list,predictions_list,labels = np.arange(n_labels))
        #ppv = cm[1,1]/(cm[1,1]+cm[0,1])
        #npv = cm[0,0]/(cm[0,0]+cm[1,0])

    #ROC_fig,auc = compute_ROC(ground_truth_list, predictions_list,encoding_dict)
    
    metrics_dict = {'accuracy':acc,'f1':f1,'precision':precision, 'recall':recall,  'loss':loss,
                    'sensitivity':sensitivity,'specificity':specificity,'confusion_matrix': cm, 
                    'ground_truth_list':ground_truth_list,'predictions_list':predictions_list}

    return metrics_dict

# def compute_ROC(ground_truth_list, predictions_list,encoding_dict):
#     # code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

#     n_labels = len(encoding_dict)
#     #print(n_labels)

#     ground_truth_list_binarized = label_binarize(ground_truth_list, classes=range(n_labels))
#     predictions_list_binarized = label_binarize(predictions_list, classes=range(n_labels))

#     #print(predictions_list_binarized.shape)

#     if n_labels == 2:
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         fpr[0], tpr[0], _ = roc_curve(ground_truth_list_binarized, predictions_list_binarized)
#         roc_auc[0] = auc(fpr[0], tpr[0])

#         auc_value = roc_auc[0]

#         fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth_list_binarized.ravel(), predictions_list_binarized.ravel())
#         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#         fig,ax = plt.subplots(figsize = (7,7))
#         fontsize = 20

#         lw = 2
#         ax.plot(fpr[0], tpr[0], color='darkorange',
#                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
#         ax.plot([0, 1], [0, 1], 'k--', lw=lw)
#         ax.set_xlim([0.0, 1.0])
#         ax.set_ylim([0.0, 1.05])
#         ax.set_xlabel('False Positive Rate',fontsize = fontsize)
#         ax.set_ylabel('True Positive Rate',fontsize = fontsize)
#         ax.set_title('Receiver Operating Characteristic',fontsize = fontsize)
#         ax.legend(loc="lower right",fontsize = fontsize)
#         ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
#         ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
#         #plt.show()
#         return fig,auc_value

#     else:

#         # Compute ROC curve and ROC area for each class
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
        
#         for i in range(n_labels):
#             fpr[i], tpr[i], _ = roc_curve(ground_truth_list_binarized[:,i], predictions_list_binarized[:,i])
#             roc_auc[i] = auc(fpr[i], tpr[i])

#         # Compute micro-average ROC curve and ROC area
#         fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth_list_binarized.ravel(), predictions_list_binarized.ravel())
#         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#         auc_value = roc_auc["micro"]
#         # First aggregate all false positive rates
#         all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_labels)]))

#         # Then interpolate all ROC curves at this points
#         mean_tpr = np.zeros_like(all_fpr)
#         for i in range(n_labels):
#             mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

#         # Finally average it and compute AUC
#         mean_tpr /= n_labels

#         fpr["macro"] = all_fpr
#         tpr["macro"] = mean_tpr
#         roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#         # Plot all ROC curves
#         lw = 2
#         fontsize = 20
#         #plt.figure()
        
#         fig,ax = plt.subplots()
#         ax.plot(fpr["micro"], tpr["micro"],
#                 label='micro-average ROC curve (area = {0:0.2f})'
#                     ''.format(roc_auc["micro"]),
#                 color='deeppink', linestyle=':', linewidth=4)

#         ax.plot(fpr["macro"], tpr["macro"],
#                 label='macro-average ROC curve (area = {0:0.2f})'
#                     ''.format(roc_auc["macro"]),
#                 color='navy', linestyle=':', linewidth=4)

#         colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#         for i, color in zip(range(n_labels), colors):
#             ax.plot(fpr[i], tpr[i], color=color, lw=lw,
#                     label='ROC curve of class {0} (area = {1:0.2f})'
#                     ''.format(encoding_dict[i], roc_auc[i]))

#         ax.plot([0, 1], [0, 1], 'k--', lw=lw)
#         ax.set_xlim([0.0, 1.0])
#         ax.set_ylim([0.0, 1.05])
#         ax.set_xlabel('False Positive Rate',fontsize = fontsize)
#         ax.set_ylabel('True Positive Rate',fontsize = fontsize)
#         ax.set_title('ROC multi-class',fontsize = fontsize)
#         ax.legend(loc="lower right",fontsize = fontsize)
#         ax.set_xticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
#         ax.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=fontsize)
#         #plt.show()
#         return fig,auc_value
    

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





        
        # save = True
        # fname = os.path.split(img_path)[1]
        # if ground == pred:
        #     if ground == 0:
        #         fname = 'tn_'+fname
        #         n_tn += 1
        #         if n_tn> N:
        #             save = False
        #     else:
        #         fname = 'tp_'+fname
        #         n_tp += 1
        #         if n_tp> N:
        #             save = False
        # else:
        #     if ground == 0:
        #         fname = 'fp_'+fname
        #         n_fp += 1
        #         if n_fp> N:
        #             save = False
        #     else:
        #         fname = 'fn_'+fname
        #         n_fn += 1
        #         if n_fn> N:
        #             save = False

        # if save:
        #     #load img
        #     image = Image.open(img_path).convert('RGB')

        #     transform = transforms.Compose([
        #     transforms.Resize((224,224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #         ])

        #     heatmap_layer = model.module.net.layer4[2].conv2
        #     #print(heatmap_layer)
        #     image_interpretable = grad_cam(model, image, heatmap_layer, transform)

        #     fig,ax = plt.subplots(1,2,figsize=(20,20))

        #     ax[0].imshow(image)
        #     #ax[0].set_title('input',fontsize=25)
        #     ax[0].axis('off')

        #     ax[1].imshow(image_interpretable)
        #     #ax[1].set_title('input',fontsize=25)
        #     ax[1].axis('off')
            
        #     plt.savefig(os.path.join(examples_path,fname))
        # else:
        #     continue




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
