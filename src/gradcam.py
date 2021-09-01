import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#implementation adapted from:
# https://github.com/tanjimin/grad-cam-pytorch-light/blob/master/grad_cam.py

class InfoHolder():

    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer

    def get_gradient(self, grad):
        self.gradient = grad

    def hook(self, model, input, output):
        output.register_hook(self.get_gradient)
        self.activation = output.detach()

def generate_heatmap(weighted_activation):
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()

def superimpose(input_img, heatmap):
    img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.6 + img * 0.4)
    pil_img = cv2.cvtColor(superimposed_img,cv2.COLOR_BGR2RGB)
    return pil_img

def to_RGB(tensor):
    tensor = (tensor - tensor.min())
    tensor = tensor/(tensor.max() + 1e-10)
    image_binary = np.transpose(tensor.numpy(), (1, 2, 0))
    image = np.uint8(255 * image_binary)
    return image

def predict_grad_cam(**kwargs):
    
    model = kwargs.get('model')
    class_index_dict = kwargs.get('class_index_dict')
    image = kwargs.get('image')
    heatmap_layer = kwargs.get('heatmap_layer')
    transform = kwargs.get('transform')
    device = kwargs.get('device')
    thres = kwargs.get('thres',0.3)
    max_pred = kwargs.get('max_pred',5)
    
    #necessary for gradcam
    info = InfoHolder(heatmap_layer)
    heatmap_layer.register_forward_hook(info.hook)

    input_tensor = transform(image).unsqueeze(0).to(device)
    output = model(input_tensor)
    
    conf_tensor,indices_tensor = torch.sort(output.data,descending=True)
    conf_arr = conf_tensor[0].cpu().numpy()
    indices_arr = indices_tensor[0].cpu().numpy()

    sorted_conf_score = [conf for conf in conf_arr if conf > thres]
    #in case all the confidences are below the threshold
    if not sorted_conf_score:
        sorted_conf_score = [conf_arr[0]]

    n_pred = min(max_pred,len(sorted_conf_score))
    sorted_labels = [index for index in indices_arr][:n_pred]
    sorted_categories = [class_index_dict[label] for label in sorted_labels]

    category_list = []
    conf_list = []
    XAI_list = []

    for label,cat,conf in zip(sorted_labels, sorted_categories, sorted_conf_score):

        output[0][label].backward(retain_graph=True)
        weights = torch.mean(info.gradient, [0, 2, 3])
        activation = info.activation.squeeze(0)
        weighted_activation = torch.zeros(activation.shape)
        for idx, (weight, activation) in enumerate(zip(weights, activation)):
            weighted_activation[idx] = weight * activation

        heatmap = generate_heatmap(weighted_activation)
        XAI = superimpose(np.asarray(image),heatmap)

        category_list.append(cat)
        conf_list.append(conf)
        XAI_list.append(XAI)

    return category_list, conf_list, XAI_list

def plot_grad_cam(**kwargs):
    
    image = kwargs.get('image')
    category_list = kwargs.get('category_list')
    confidence_list = kwargs.get('confidence_list')
    XAI_list = kwargs.get('XAI_list')
    ground_truth = kwargs.get('ground_truth')
    saving_path = kwargs.get('saving_path')
    fontsize = kwargs.get('fontsize',10)
    figsize = kwargs.get('figsize',(15,15))
    
    fig,ax = plt.subplots(1,len(category_list)+1,figsize = figsize)
    ax[0].imshow(image)
    ax[0].axis('off')
    if ground_truth:
        ax[0].set_title(ground_truth)
        
    for i, (cat, conf, XAI) in enumerate(zip(category_list, confidence_list, XAI_list)):
        ax[i+1].imshow(XAI)
        ax[i+1].set_title(f'{cat}, {conf:.3f}',size=fontsize)
        ax[i+1].axis('off')
        
    if saving_path:
        plt.savefig(saving_path)