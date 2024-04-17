import os
import torch
from cqt_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *

# multi_size train
def multi_train(**kwargs):
    parallel = True 
    opt.model = 'CQTNet'
    opt.notes='CQTNet'
    opt.batch_size=32
    #opt.load_latest=True
    #opt.load_model_path = ''
    opt._parse(kwargs)
    
    # step1: configure model
    
    model = getattr(models, opt.model)() 
        
    if parallel is True: 
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    
    # Change network model - Fine Tuning

    
    
    # To device
    
    model.to(opt.device)
    print(model)
    
    # step2: data
   
    train_data1 = CQT('train', out_length=300)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)

    train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1)

    #step3: criterion and optimizer
    
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=2, verbose=True,min_lr=5e-6)
    
    #train
    best_MAP=0
    
    val_slow(model, val_dataloader, -1)
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for data, label in tqdm(train_dataloader1):
            
            # train model
            input_data = data.requires_grad_()
            input_data = input_data.to(opt.device)
            
            target = label.to(opt.device)

            optimizer.zero_grad()
            score, _ = model(input_data)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num += target.shape[0]
            
        running_loss /= num 
        
        print(running_loss)
        
        if parallel is True:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        
        # update learning rate
        scheduler.step(running_loss) 
        
        # validate
        MAP=0
        MAP += val_slow(model, val_dataloader, epoch)
                
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
        print('')
        model.train()

   
@torch.no_grad()
def val_slow(model, dataloader, epoch):
    
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
        input_data = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input_data)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    features = norm(features)

    dis2d = -np.matmul(features, features.T) # [-1,1] Because normalized, so mutmul is equal to ED
    np.save('dis.npy',dis2d)
    np.save('label.npy',labels)

    MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(epoch, MAP, top10, rank1 )
    
    model.train()
    
    return MAP

def test(**kwargs):
    opt.batch_size=1
    opt.num_workers=1
    opt.model = 'CQTNet'
    opt.load_latest = False
    opt.load_model_path = 'check_points/CQTNet.pth'
    opt._parse(kwargs)
    
    model = getattr(models, opt.model)() 
    #print(model)
    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    
    val_slow(model, val_dataloader, 0)


if __name__=='__main__':
    import fire
    fire.Fire()
