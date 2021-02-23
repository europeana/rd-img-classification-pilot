import os
import requests
import json
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transformsa

from torch_utils import ResNet

from gradcam import grad_cam

#to do: refactor this code for the colab notebook

# Europeana API

class EuropeanaAPI:
  def __init__(self,wskey):
    self.wskey = wskey

  def search(self,query,n=20):
    
    CHO_list = []
    response = {'nextCursor':'*'}
    while 'nextCursor' in response:
      params = { 'reusability':'open','media':True,'cursor':response['nextCursor'] ,'qf':'TYPE:IMAGE', 'query':query, 'wskey':'api2demo'}
      response = requests.get('https://www.europeana.eu/api/v2/search.json', params = params).json()
      CHO_list += response['items']
      if len(CHO_list)>n:
        CHO_list = CHO_list[:n]
        break

    return CHO_list


def img_from_CHO(CHO):
  try:
    URL = CHO['edmIsShownBy'][0]
    url_response = requests.get(URL)
    img = Image.open(BytesIO(url_response.content)).convert('RGB')
    return img
  except:
    return None

def load_pytorch_model(device,root_path):

    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #root_path = os.path.split(dir_path)[0]

    model = ResNet(34,20)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(root_path,'checkpoint.pth'),map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(root_path,'checkpoint.pth')))

    model = model.to(device)
    model.eval()

    with open(os.path.join(root_path,'class_index.json'),'r') as f:
        class_index_dict = json.load(f)

    return model, class_index_dict

def make_prediction(model,img,device):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    heatmap_layer = model.net.layer4[1].conv2
    image_interpretable,idx_pred,conf = grad_cam(model, img, heatmap_layer, transform, device)

    return idx_pred.item(), conf, image_interpretable

def plot_prediction(img,XAI_img):

    fig,ax = plt.subplots(1,2,figsize=(20,20))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(XAI_img)
    ax[1].axis('off')
    plt.show()
    return 
    