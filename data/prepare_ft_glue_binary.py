from datasets import load_dataset
import torch
import sentencepiece as spm
import os
tokenizer = spm.SentencePieceProcessor(model_file='/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model')
from tqdm import tqdm
import pandas as pd

TASK = '/sst2'
#TASK = '/CoLA'
LANG = 'en'
CONTEXT_LEN = 256

# data = pd.read_csv(f'glue{TASK}/train.tsv',sep='\t',header=None)
# print(data.columns)
# print(data[:2])
INP_INDEX = 'sentence'
OUTP_INDEX = 'label'
data = load_dataset('glue',name=TASK[1:].lower(),revision='b4b3e3965b74b673e8a6528d1bc059d7a77b53ff')

if not os.path.exists(f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/')

if not os.path.exists(f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/')

train = []
train_mask = []


for row in data['train']:
    inp = row[INP_INDEX]
    outp = row[OUTP_INDEX]
    train.append(inp)
    train_mask.append(outp)


train = tokenizer.encode(train)

temp = []

for i in tqdm(range(len(train))):
    temp.append([1] + train[i] + [2])

train = temp


clip = 0
total = 0
for i in tqdm(range(len(train))):
    total +=1 
    if len(train[i])>(CONTEXT_LEN):
        clip+=1
        train[i] = train[i][:CONTEXT_LEN]
    else:
        train[i] = train[i] + [3]*(CONTEXT_LEN-len(train[i]))
    assert len(train[i])==CONTEXT_LEN
print(clip,total)

train = torch.IntTensor(train)
train_mask = torch.IntTensor(train_mask).view(-1,1)
print(torch.sum(train_mask)/train_mask.shape[0])

print(train.shape, train_mask.shape)

train = torch.cat([train,train_mask],dim=1)
print(train.shape)
torch.save(train,f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/train.pt')






train = []
train_mask = []


#data = pd.read_csv(f'glue{TASK}/dev.tsv',sep='\t')

for row in data['validation']:
    #print(row)
    inp = row[INP_INDEX]
    outp = row[OUTP_INDEX]
    train.append(inp)
    train_mask.append(outp)


train = tokenizer.encode(train)

temp = []

for i in tqdm(range(len(train))):
    temp.append([1] + train[i] + [2])

train = temp


clip = 0
total = 0
for i in tqdm(range(len(train))):
    total +=1 
    if len(train[i])>(CONTEXT_LEN):
        clip+=1
        train[i] = train[i][:CONTEXT_LEN]
    else:
        train[i] = train[i] + [3]*(CONTEXT_LEN-len(train[i]))
    assert len(train[i])==CONTEXT_LEN
print(clip,total)

train = torch.IntTensor(train)
train_mask = torch.IntTensor(train_mask).view(-1,1)
print(torch.sum(train_mask)/train_mask.shape[0])

print(train.shape, train_mask.shape)

train = torch.cat([train,train_mask],dim=1)
print(train.shape)
torch.save(train,f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/val.pt')

