import json
import fire
from pathlib import Path

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import lightly
import lightly.utils.io as io

from ss_models import MoCoModel
from ss_models import MoCoModel_benchmarking

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm


#from utilities import load_self_supervised

def main(**kwargs):

    data_path = kwargs.get('data_dir')
    pretrained_dir = kwargs.get('pretrained_dir')
    saving_path = kwargs.get('saving_path')
    input_size = kwargs.get('input_size',224)
    batch_size = kwargs.get('batch_size',64)
    num_workers = kwargs.get('num_workers',4)
    sample = kwargs.get('sample',1.0)

    pretrained_dir = Path(pretrained_dir)
    saving_path = Path(saving_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # *** Load data *** 

    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=lightly.data.collate.imagenet_normalize['mean'],
                std=lightly.data.collate.imagenet_normalize['std'],
            )
    ])

    dataset = lightly.data.LightlyDataset(
        input_dir=data_path,
        transform = transform
    )
    
    #idx = np.random.randint(len(dataset),size = int(len(dataset)*sample))
    #dataset = torch.utils.data.Subset(dataset,idx)

    dataloader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers
    )
    

    # *** Load model *** 

    with open(pretrained_dir.joinpath('conf.json')) as f:
        conf = json.load(f)

    ss_model = conf['ss_model']

    print(conf)

    resnet_dict = {
        18: torchvision.models.resnet18(pretrained=False),
        34: torchvision.models.resnet34(pretrained=False),
        50: torchvision.models.resnet50(pretrained=False),
        101: torchvision.models.resnet101(pretrained=False),
    }

    backbone = resnet_dict[conf['model_size']]

    if conf['benchmarking']:
        model = MoCoModel_benchmarking(backbone,None,1,num_ftrs=conf['num_ftrs'],).to(device)
    else:
        model = MoCoModel(backbone,num_ftrs=conf['num_ftrs']).to(device)

    model.backbone.load_state_dict(torch.load(pretrained_dir.joinpath('checkpoint.pth')))


    # *** Compute embeddings *** 

    print('Computing embeddings...')
    embeddings = []
    filenames = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, (x, _, fnames) in tqdm(enumerate(dataloader_train)):
            x = x.to(device)
            y = model.backbone(x)
            y = y.squeeze()
            embeddings.append(y)
            filenames = filenames + list(fnames)

    
    
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().numpy()
    #print(embeddings.shape)
    print('Finished computing embeddings')
    
    # dummy and necessary list
    labels = [None for fname in filenames]

    io.save_embeddings(
        saving_path,
        embeddings,
        labels,
        filenames)


if __name__ == '__main__':
    fire.Fire(main)
