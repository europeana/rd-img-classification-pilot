from notebook_env import *
from data_wrangling import *
import models
import time
import sklearn
import torch
import datetime

#to do: document



def path2DataFrame(data_dir):
    """reads a directory structured in classes into a dataframe"""
    category_list = os.listdir(data_dir)
    img_path_list = []
    img_category_list = []
    for cat in category_list:
        cat_path = os.path.join(data_dir,cat)
        img_cat_list = [os.path.join(cat_path,filename) for filename in os.listdir(cat_path)]
        img_path_list += img_cat_list
        img_category_list += [cat for i in range(len(img_cat_list))]
        
    return pd.DataFrame({'file_path':img_path_list,'category':img_category_list})

def path2DataFrame_unlabeled(data_dir):
    img_path_list = [os.path.join(data_dir,filename) for filename in os.listdir(data_dir)]
    return pd.DataFrame({'file_path':img_path_list})

def drop_categories_df(df,categories):
    """categories can be str or list of str"""
    if isinstance(categories,str):
        categories = [categories]
    def filter(x):
        return x not in categories
    # apply filter
    df_ = df[df['category'].map(filter)]
    # re-index the dataframe
    df_.index = np.arange(df_.shape[0])
    return df_

def downsample_df(df, cat_name, size):

    rem_df = df.loc[df['category'].map(lambda x: x != cat_name)]
    cat_df = df.loc[df['category'].map(lambda x: x == cat_name)]
    cat_df = cat_df.iloc[np.random.randint(0,cat_df.shape[0],size)]
    df = pd.concat((rem_df,cat_df))
    return df

def get_class_weights(y_encoded,encoding_dict):
    """Calculates the weights for the Cross Entropy loss """
    data_dict = get_imgs_per_cat(y_encoded)       
    N = sum(data_dict.values())
    print('Percentage of images in each category:\n')              
    #calculate weights as the inverse of the frequency of each class
    weights = []
    for k in range(len(data_dict)):
        v = data_dict[k]
        weights.append(N/v)
        print('{}: {:.6g} %'.format(encoding_dict[k],100.0*v/N))    
    print('Weights: {}\n'.format(weights))
    print('\n')      
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

def time_stamp():
    return str(datetime.datetime.now()).replace(' ','_')[:-7]

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

        # ts = time_stamp()
        # experiment_id = ts.replace('-','').replace(':','')
        filename = 'training_info.pth'
        info_file_path = os.path.join(dest_path,filename)
        torch.save(self.info, info_file_path)


# to do: include into Experiment class
def save_experiment(**kwargs):

    #to do: model name

    time_train = kwargs.get('time_train',0)
    epochs = kwargs.get('epochs',0)
    learning_rate = kwargs.get('learning_rate',None)
    trainloader = kwargs.get('trainloader',None)
    testloader = kwargs.get('testloader',None)

    execution_path = kwargs.get('execution_path',None)

    experiment_id = kwargs.get('experiment_id',None)

    if not experiment_id:
        #date and time, 
        ts = time_stamp()
        experiment_id = ts.replace('-','').replace(':','')

    
    #make directory for saving the experiment
    experiment_path = os.path.join(self.execution_path, f'training_{experiment_id}')
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    
    
    data_dict = { 
                'trainloader': trainloader,
                'testloader': testloader,

                'encoding_dict': self.encoding_dict,
                'weights': self.weights,

                'epochs': epochs,
                'model': self.net,
                'optimizer': self.optimizer,
                'loss_function': self.criterion,
                'learning_rate': learning_rate,
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,

                'training_time_minutes':time_train,
                'loss_train_list':self.loss_train_list,
                'loss_test_list':self.loss_test_list,
                'acc_test_list':self.acc_test_list,
                }
    
    filename = f'training_info_{experiment_id}.pth'
    info_file_path = os.path.join(experiment_path,filename)
    print(info_file_path)
    torch.save(data_dict, info_file_path)
    



    # 'img_aug': self.img_aug,

    # info_file_path = os.path.join(experiment_path,f'training_info_{experiment_id}.json')
    # with open(info_file_path,'w') as f:
    #     json.dump(data_dict,f)


    #train and test data
    #metrics: accuracy, confusion matrix

    return 



