import torch
import sklearn
import fire
from pathlib import Path
from torch_utils import ResNet, path2DataFrame, label_encoding, get_class_weights, TrainingDataset, evaluate, collate_fn
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import imblearn
import json

from sklearn.metrics import classification_report

from sklearn import preprocessing


from ds_utils import create_dir

def main(**kwargs):
    results_path = kwargs.get('results_path') 
    saving_dir = kwargs.get('saving_dir') 
    data_path = kwargs.get('data_path') 
    batch_size = kwargs.get('batch_size',16) 
    num_workers = kwargs.get('num_workers',4) 

    saving_dir = Path(saving_dir)

    create_dir(saving_dir)

    results_path = Path(results_path)

    metrics_list = ['accuracy','precision','recall','sensitivity','specificity','f1','confusion_matrix','report']

    metrics_dict = {m:[] for m in metrics_list}


    # get list of files
    file_list = [x for x in results_path.iterdir() if not x.is_dir()]
    split_list = [x for x in results_path.iterdir() if x.is_dir()]

    if file_list:
        split_list = [results_path]

    for split_path in split_list:

        model_path = split_path.joinpath('checkpoint.pth')

        #load config
        conf_path = split_path.joinpath('conf.json')
        with open(conf_path,'r') as f:
            conf = json.load(f)

        #load class_index_dict
        with open(split_path.joinpath('class_index.json'),'r') as f:
            class_index_dict = json.load(f)

        input_size = conf['input_size']
        resnet_size = conf['resnet_size']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        df = path2DataFrame(data_path)    

        X = df['file_path'].values
        y = df['category'].values

        le = preprocessing.LabelEncoder()
        le.classes_ = [class_index_dict[k] for k in class_index_dict.keys()]
        y = le.transform(y)

        n_classes = len(class_index_dict)

        weights = get_class_weights(y,class_index_dict)
        loss_function = nn.CrossEntropyLoss(reduction ='sum',weight=torch.FloatTensor(weights).to(device))   

        test_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        test_dataset = TrainingDataset(X,y,transform=test_transform)
        testloader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers,
            drop_last=True,
            collate_fn = collate_fn)

        #initialize model
        model = ResNet(resnet_size,n_classes).to(device)
        model.load_state_dict(torch.load(model_path))

        split_metrics_dict, ground_truth_list, predictions_list, test_images_list = evaluate(
            model = model,
            dataloader = testloader,
            device = device,
            loss_function = loss_function
            )

        
        target_names = [class_index_dict[k] for k in class_index_dict.keys()]
        report = classification_report(ground_truth_list, predictions_list, target_names=target_names,output_dict=True)

        split_metrics_dict.update({'report':report})

        for k,v in split_metrics_dict.items():
            if k not in ['loss']:
                metrics_dict[k].append(v)

    for k in metrics_dict.keys():
        if k == 'confusion_matrix':
            cm = np.zeros((n_classes,n_classes))
            for _cm in metrics_dict[k]:
                cm += _cm
            cm = cm/len(metrics_dict[k])
            metrics_dict[k] = cm.astype(int)

        elif k == 'report':

            cat_metrics = {class_index_dict[k]:{'precision':[],'recall':[],'f1-score':[],} for k,v in class_index_dict.items()}
            for report in metrics_dict[k]:
                for category in cat_metrics.keys():
                    cat_metrics[category]['precision'].append(report[category]['precision'])
                    cat_metrics[category]['recall'].append(report[category]['recall'])
                    cat_metrics[category]['f1-score'].append(report[category]['f1-score'])

            for cat in cat_metrics.keys():
                cat_metrics[cat]['precision'] = {'mean':np.mean(cat_metrics[cat]['precision']),'std':np.std(cat_metrics[cat]['precision'])}
                cat_metrics[cat]['recall'] = {'mean':np.mean(cat_metrics[cat]['recall']),'std':np.std(cat_metrics[cat]['recall'])}
                cat_metrics[cat]['f1-score'] = {'mean':np.mean(cat_metrics[cat]['f1-score']),'std':np.std(cat_metrics[cat]['f1-score'])}

            metrics_dict.update({'report':cat_metrics})


        else:
            metrics_dict[k] = {'mean':np.mean(metrics_dict[k]),'std':np.std(metrics_dict[k])}
            print(k,metrics_dict[k]['mean'],'+-',metrics_dict[k]['std'])

    torch.save(metrics_dict,saving_dir.joinpath('evaluation_results.pth'))

if __name__ == '__main__':
    fire.Fire(main)