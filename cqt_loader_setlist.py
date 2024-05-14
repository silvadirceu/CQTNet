import os
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import deepdish as dd 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from typing import Union
import PIL
import glob
from tqdm import tqdm


def cut_data(data, out_length: Union[int, None] = None):
    
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:out_length+offset, :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
            
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            data = data[:out_length, :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


def shorter(feature, mean_size=2):
    length, height  = feature.shape
    new_f = np.zeros((int(length/mean_size),height),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_f[i,:] = feature[i*mean_size:(i+1)*mean_size,:].mean(axis=0)
    return new_f

def change_speed(data, l=0.7, r=1.5): # change data.shape[0]
    new_len = int(data.shape[0]*np.random.uniform(l,r))
    maxx = np.max(data)+1
    data0 = PIL.Image.fromarray((data*255.0/maxx).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(size=(new_len,data.shape[1]))
    ])
    new_data = transform(data0)
    return np.array(new_data)/255.0*maxx

def SpecAugment(data):
    F = 24
    f = np.random.randint(F)
    f0 = np.random.randint(84-f)
    data[f0:f0+f,:]*=0
    return data


#######################################################
#               Define Dataset Class
#######################################################

class CQT(Dataset):
    def __init__(self, work_path, mode=False, out_length=200, num_workers=1):
        
        self.works_paths = [] #to store image paths in list
        self.classes = [] #to store class values
        self.out_length = out_length
        
        for data_path in glob.glob(work_path + '/*'):
            self.classes.append(data_path.split('/')[-1])
            self.works_paths.append(glob.glob(data_path + '/*'))
            
        self.works_paths = list(flatten(self.works_paths))
        
        self.idx_to_class = {i:j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
        
        if mode == "train":
            self.transform = transforms.Compose([
            lambda x: SpecAugment(x), #SpecAugment 频谱增强一次
            lambda x: SpecAugment(x), #SpecAugment 频谱增强 x 2
            lambda x : x.T,
            lambda x : change_speed(x, 0.7, 1.3), # 速度随机变化
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
        ])
        else:
            self.transform = transforms.Compose([
            lambda x : x.T,
            #lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x))+ 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1,0).unsqueeze(0),
            ])
            
                    
    def get_idx_to_class(self):
        return self.idx_to_class
    
    def get_class_to_idx(self):
        return self.class_to_idx    
    
    def get_data_params(self, idx):
        filepath = self.works_paths[idx]
        data = dd.io.load(filepath)
        
        return data["params"]
    
    def get_nr_classes(self):
        return len(self.classes)
    
    def get_labels(self, idx):
        filepath = self.works_paths[idx]
        label, track = filepath.split('/')[-2], filepath.split('/')[-1]
        
        return label, track
    
    def get_transform(self):
        return self.transform 
    
    def __len__(self):
        return len(self.works_paths)

    def __getitem__(self, idx):
        filepath = self.works_paths[idx]
        data = dd.io.load(filepath)
        
        label = filepath.split('/')[-2]
        label = self.class_to_idx[label]
        #set_id= data["label"]
        #print(f"filepath {label} -- hr_file {self.class_to_idx[set_id]}")
        
        cqt = self.transform(data["cqt"])
            
        return cqt, label

    

