import os
import sys
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from skimage.io import imsave
import pandas as pd

from shutil import copyfile
# set pandas display options
import  pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

import sys
sys.path.append('/home/jcejudo/rd-img-classification-pilot/src')