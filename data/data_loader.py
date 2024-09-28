import torch

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities import CombinedLoader

import lightning.pytorch as pl
import random
import os

# class CustomRandomSampler(Sampler):
#     def __init__(self, data, context_length, replacement=False, num_samples=None):
#         self.data = data
#         self.context_length = context_length
#         self.replacement = replacement
#         self._num_samples = num_samples
        
#     @property
#     def num_samples(self):
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self):
#         n = len(self.data)
#         if self.replacement:
#             return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
#         return iter(torch.randperm(n).tolist())

#     def __len__(self):
#         return self.num_samples
    
class TextDataset(Dataset):
    def __init__(self, text_file, config, tokenizer=None,num_samples=None):
        with open(text_file) as f:
            lines = f.readlines()
        random.Random(config.SHUFFLE_SEED).shuffle(lines)
        if num_samples!=None:
            lines = lines[:num_samples]
        self.text_file = lines
        self.file_size_ = len(self.text_file)
        print('length of text file', len(self.text_file))
        self.config = config
        self.tokenizer = tokenizer 
        self.pad_reference = torch.zeros(self.config.context_len+1) + self.tokenizer.piece_to_id('<pad>')
        
    def __getitem__(self, idx):
        #print(idx)
        current_window = self.text_file[idx]
        tokenized_output = self.tokenizer.encode(current_window)
        tokenized_output = [self.tokenizer.piece_to_id('<s>')] + tokenized_output + [self.tokenizer.piece_to_id('</s>')]
        tokenized_output = torch.tensor(tokenized_output)
        #print(tokenized_output.shape)
        if tokenized_output.shape[0]<self.config.context_len+1:
            pad_reference = self.pad_reference
            pad_reference[:len(tokenized_output)] = tokenized_output
            tokenized_output = pad_reference
        #print(tokenized_output.shape)
        #assert tokenized_output.shape[0]>=self.config.context_len+1
        text = tokenized_output[0:self.config.context_len].long()
        target = tokenized_output[1:self.config.context_len+1].long()
        #print(text.shape,target.shape)
        #assert text.shape == target.shape
        return (text, target)

    def __len__(self):
        return self.file_size_
    

class TokenizedDataset(Dataset):
    def __init__(self, text_file, config, tokenizer=None):
        self.text_file = None
        if type(text_file)==list:
            files = []
            for file in text_file:
                files.append(torch.load(file))
            self.text_file = torch.cat(files,dim=0)
            self.text_file = self.text_file[torch.randperm(self.text_file.size()[0])].long()
            print(self.text_file.shape)
        else:
            self.text_file = torch.load(text_file)
        self.file_size_ = self.text_file.shape[0]
        print('length of text file', self.text_file.shape)
        self.config = config
        self.tokenizer = tokenizer 
        
    def __getitem__(self, idx):
        tokenized_output = self.text_file[idx,:]
        text = tokenized_output[0:self.config.context_len]
        target = tokenized_output[1:self.config.context_len+1]
        return (text, target)

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
    
