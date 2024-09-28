import torch

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities import CombinedLoader

import lightning.pytorch as pl
import random
import os

class TokenizedDataset(Dataset):
    def __init__(self, text_file, config, tokenizer=None):
        self.text_file = None
        if type(text_file)==list:
            files = []
            masks = []
            for file in text_file:
                files.append(torch.load(file))
                masks.append(torch.load(file+'.mask'))
            self.text_file = torch.cat(files,dim=0)
            self.mask_file = torch.cat(masks,dim=0)
            #self.text_file = self.text_file[torch.randperm(self.text_file.size()[0])].long()
            print(self.text_file.shape)
            print('Mask shape',self.mask_file.shape)
        else:
            self.text_file = torch.load(text_file)
            self.mask_file = torch.load(text_file+'.mask')
            print('Mask shape',self.mask_file.shape)
        self.file_size_ = self.text_file.shape[0]
        print('length of text file', self.text_file.shape)
        self.config = config
        self.tokenizer = tokenizer 
        
    def __getitem__(self, idx):
        tokenized_output = self.text_file[idx,:]
        mask = self.mask_file[idx,:]
        text = tokenized_output[0:self.config.context_len]
        target = tokenized_output[1:self.config.context_len+1]
        return (text, target, mask)

    def __len__(self):
        return self.file_size_


class IndicGPTDataModule(pl.LightningDataModule):
        def __init__(self, config=None, tokenizer=None, train_file: str = "path/to/dir", val_file: list = ["path/to/dir"], batch_size: int = 32):
            super().__init__()
            self.config = config
            self.tokenizer = tokenizer
            self.train_dir = train_file
            self.val_dir = val_file
            self.batch_size = batch_size

        def setup(self, stage: str):
            self.train_data = TokenizedDataset(self.train_dir,self.config,self.tokenizer)
            self.val_data = [TokenizedDataset(self.val_dir[i],self.config,self.tokenizer) for i in range(len(self.val_dir))]

        def train_dataloader(self):
            data = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.config.num_workers,pin_memory=self.config.PIN_MEMORY)
            return data

        def val_dataloader(self):
            data = [DataLoader(self.val_data[i], batch_size=self.batch_size,num_workers=self.config.num_workers,pin_memory=self.config.PIN_MEMORY) for i in range(len(self.val_data))]
            return data
    
