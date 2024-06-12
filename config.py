# coding:utf8
import warnings
import torch


def select_device(use_gpu=False, gpu_device="cuda:1"):
    #verificar versao de gpu
    print('torch.cuda.is_available()', torch.cuda.is_available())
    print('use_gpu', use_gpu)
    if torch.cuda.is_available() and use_gpu:
        device = gpu_device
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = "cpu"
        torch.set_default_tensor_type('torch.FloatTensor')
        
    print(torch.device(device))
    return torch.device(device)


class DefaultConfig(object):
    
    def __init__(self, 
                 load_model_path, 
                 notes, 
                 batch_sz=128, 
                 max_epoch=100, 
                 lr=0.001, 
                 lr_decay=0.8, 
                 weight_decay=1e-5, 
                 use_gpu=True, 
                 num_workers=4, 
                 parallel=True, 
                 load_latest=False,
                 gpu_device="cuda:0") -> None:
        
        self.notes = notes
        self.batch_sz = batch_sz
        self.model = 'CQTNet'  
        self.feature = 'cqt'
        self.load_model_path = load_model_path  
        self.load_latest = load_latest
        self.batch_size = batch_sz  # batch size
        self.use_gpu = use_gpu  # user GPU or not
        self.num_workers = num_workers  # how many workers for loading data
        self.parallel = parallel
        self.max_epoch = max_epoch
        self.lr = lr  # initial learning rate
        self.lr_decay = lr_decay  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = weight_decay  
        self.device = select_device(use_gpu=use_gpu, gpu_device=gpu_device)

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('+------------------------------------------------------+')
        print('|', 'user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('|', k, getattr(self, k))
        print('+------------------------------------------------------+')
        
