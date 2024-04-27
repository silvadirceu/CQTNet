import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect
import torchvision
import PIL
import deepdish as dd
import glob
from typing import Union


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

class CQT(Dataset):
    def __init__(self, mode='train', out_length: Union[int, None] = None):
        
        self.indir = '/Users/dirceusilva/Documents/BancoDados/setlist_all/setlist_65k/features/universe_develop/'
        
        self.mode = mode
        
        if mode == 'train': 
            self.filepath = os.path.join(self.indir, "universe_train")
            self.file_list = glob.glob(os.path.join(self.filepath, "**/*.h5"), recursive=True)
        elif mode == 'val':
            self.filepath = os.path.join(self.indir, "universe_val")
            self.file_list = glob.glob(os.path.join(self.filepath, "**/*.h5"), recursive=True)
        elif mode == 'test': 
            self.filepath = os.path.join(self.indir, "universe_test")
            self.file_list = glob.glob(os.path.join(self.filepath, "**/*.h5"), recursive=True)
            
        print(len(self.file_list))
        self.list_setid = set([a.split("/")[-2] for a in self.file_list])
        
        # with open(filepath, 'r') as fp:
        #     self.file_list = [line.rstrip() for line in fp]
            
        self.out_length = out_length
        
    def get_setids(self):
        return self.list_setid
    
    def get_setids_len(self):
        return len(self.list_setid)
    
    def get_filepath(self):
        return self.filepath
    
    def get_filelist(self):
        return self.file_list

    def __getitem__(self, index):
        
        transform_train = transforms.Compose([
            lambda x : x.T,
            #lambda x : change_speed(x, 0.7, 1.3),
            #lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x : cut_data(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1, 0).unsqueeze(0),
        ])
        
        transform_test = transforms.Compose([
            lambda x : x.T,
            #lambda x : x-np.mean(x),
            lambda x : x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x : cut_data_front(x, self.out_length),
            lambda x : torch.Tensor(x),
            lambda x : x.permute(1, 0).unsqueeze(0),
        ])
        
        filepath = self.file_list[index]
        data = dd.io.load(filepath)
        set_id, version_id = data["label"], data["track_id"]

         # from 12xN to Nx12 and cut data
        if self.mode == 'train':
            data = transform_train(data["cqt"])
        else:
            data = transform_test(data["cqt"])
            
        return data, int(set_id)

    def __len__(self):
        return len(self.file_list)

    
if __name__=='__main__':
    train_dataset = HPCP('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
