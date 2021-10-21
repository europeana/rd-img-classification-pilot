from pathlib import Path
from PIL import Image
import numpy as np
import fire
import json

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

import lightly

from ss_models import MoCoModel_benchmarking, BYOLModel_benchmarking,MoCoModel, BYOLModel

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(**kwargs):

    data_path = kwargs.get('data_path',None)
    train_knn_data_path = kwargs.get('train_knn_data_path',None)
    test_knn_data_path = kwargs.get('test_knn_data_path',None)
    saving_dir = kwargs.get('saving_dir',None)

    knn_k = kwargs.get('knn_k',200)
    max_epochs = kwargs.get('max_epochs',200)
    model_size = kwargs.get('model_size',18) # 18, 34, 50, 101, 152
    learning_rate = kwargs.get('learning_rate',0.01)
    input_size = kwargs.get('input_size',224)
    num_ftrs = kwargs.get('num_ftrs',512)
    batch_size = kwargs.get('batch_size',64)
    num_workers = kwargs.get('num_workers',8)
    memory_bank_size = kwargs.get('memory_bank_size',4096)
    sample = kwargs.get('sample',1.0)
    ss_model = kwargs.get('ss_model','moco')

    hf_prob = kwargs.get('hf_prob',0.0) # horizontal flip
    vf_prob = kwargs.get('vf_prob',0.0) # vertical flip
    cj_prob = kwargs.get('cj_prob',0.0) # color jitter 
    gb_prob = kwargs.get('gb_prob',0.0) # gaussian blur 
    
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)
    
    conf = {
        'num_ftrs':num_ftrs,
        'max_epochs':max_epochs,
        'learning_rate':learning_rate,
        'input_size':input_size,
        'batch_size':batch_size,
        'hf_prob':hf_prob,
        'vf_prob':vf_prob,
        'gb_prob':gb_prob,
        'cj_prob':cj_prob,
        'ss_model':ss_model,
        'model_size':model_size,
        'benchmarking':True,
        }

    print('Configuration:')
    for k,v in conf.items():
        print(f'{k}: {v}')

    with open(saving_dir.joinpath('conf.json'),'w') as f:
        json.dump(conf,f)


    # *** Image augmentation ***

    color_jitter = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)

    kernel_size = 3
    gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.0)),
        torchvision.transforms.RandomApply([color_jitter], p=cj_prob),
        torchvision.transforms.RandomApply([gaussian_blur], p=gb_prob),
        #torchvision.transforms.Resize((input_size,input_size)),
        torchvision.transforms.RandomHorizontalFlip(p=hf_prob),
        torchvision.transforms.RandomVerticalFlip(p=vf_prob),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])

    # *** Loading data ***

    #collate_fn = lightly.data.BaseCollateFunction(train_transforms)

    collate_fn = lightly.data.ImageCollateFunction(
        input_size=input_size,
        # require invariance to flips and rotations
        hf_prob=0.5,
        #vf_prob=0.5,
        #rr_prob=0.5,
        # satellite images are all taken from the same height
        # so we use only slight random cropping
        min_scale=0.5,
        # use a weak color jitter for invariance w.r.t small color changes
        cj_prob=0.1,
        cj_bright=0.1,
        cj_contrast=0.1,
        cj_hue=0.1,
        cj_sat=0.1,
        normalize = lightly.data.collate.imagenet_normalize
    )
    
    dataset_train = lightly.data.LightlyDataset(
        input_dir=data_path
    )

    idx = np.random.randint(len(dataset_train),size = int(sample*len(dataset_train)))
    dataset_train = torch.utils.data.Subset(dataset_train,idx)

    print(len(dataset_train))

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    # test_collate_fn = lightly.data.BaseCollateFunction(test_transforms)

    # dataset_knn_train = lightly.data.LightlyDataset(
    #     input_dir=train_knn_data_path,
    #     transform = train_transforms
    # )

    # dataset_knn_test = lightly.data.LightlyDataset(
    #     input_dir=test_knn_data_path,
    #     transform = test_transforms
    # )

    # dataloader_knn_train = torch.utils.data.DataLoader(
    #     dataset_knn_train,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=num_workers
    # )

    # dataloader_knn_test = torch.utils.data.DataLoader(
    #     dataset_knn_test,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=num_workers
    # )

    # *** Training ***

    gpus = [0] if torch.cuda.is_available() else 0

    #num_classes = len(list(test_knn_data_path.iterdir()))

    resnet_dict = {
        18: torchvision.models.resnet18(pretrained=False),
        34: torchvision.models.resnet34(pretrained=False),
        50: torchvision.models.resnet50(pretrained=False),
        101: torchvision.models.resnet101(pretrained=False),
    }

    backbone = resnet_dict[model_size]

    if ss_model == 'moco':

        # model = MoCoModel_benchmarking(
        #     backbone,
        #     dataloader_knn_train,
        #     num_classes,
        #     num_ftrs = num_ftrs,
        #     knn_k=knn_k,
        #     learning_rate=learning_rate,
        #     max_epochs=max_epochs,
        #     )

        model = MoCoModel(
            backbone,
            num_ftrs = num_ftrs,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            )
            
    elif ss_model == 'byol':

        # model = BYOLModel_benchmarking(
        #     backbone,
        #     dataloader_knn_train,
        #     num_classes,
        #     num_ftrs = num_ftrs,
        #     knn_k=knn_k,
        #     learning_rate=learning_rate,
        #     max_epochs=max_epochs,
        #     )

        model = BYOLModel(
            backbone,
            num_ftrs = num_ftrs,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            )

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        gpus=gpus,
        progress_bar_refresh_rate=100)

    trainer.fit(
        model,
        train_dataloader = dataloader_train,
        #val_dataloaders = dataloader_knn_test,
    )

    print('Finished training')

    #print('max accuracy: ',model.max_accuracy)
    torch.save(model.backbone.state_dict(),saving_dir.joinpath('checkpoint.pth'))


# def main(**kwargs):

#     data_path = kwargs.get('data_path',None)
#     train_knn_data_path = kwargs.get('train_knn_data_path',None)
#     test_knn_data_path = kwargs.get('test_knn_data_path',None)

#     saving_dir = kwargs.get('saving_dir',None)
#     max_epochs = kwargs.get('max_epochs',200)
#     learning_rate = kwargs.get('learning_rate',0.1)
#     input_size = kwargs.get('input_size',224)
#     num_ftrs = kwargs.get('num_ftrs',512)
#     batch_size = kwargs.get('batch_size',64)
#     hf_prob = kwargs.get('hf_prob',0.0) # horizontal flip
#     vf_prob = kwargs.get('vf_prob',0.0) # vertical flip
#     memory_bank_size = kwargs.get('memory_bank_size',4096)
#     model_size = kwargs.get('model_size',18)
#     knn_k = kwargs.get('knn_k',100)
#     sample = kwargs.get('sample',1.0)

#     if not data_path:
#         raise Exception('data_path not provided')

#     if not saving_dir:
#         raise Exception('saving_dir not provided')

#     train(
#         data_path = data_path,
#         train_knn_data_path = train_knn_data_path,
#         test_knn_data_path = test_knn_data_path,
#         knn_k = knn_k,
#         saving_dir = saving_dir,
#         max_epochs = max_epochs,
#         learning_rate = learning_rate,
#         input_size = input_size,
#         num_ftrs = num_ftrs,
#         batch_size = batch_size,
#         hf_prob = hf_prob,
#         vf_prob = vf_prob,
#         model_size = model_size,
#         memory_bank_size = memory_bank_size,
#         sample = sample
#         )

if __name__=="__main__":

    fire.Fire(main)
