from datasets import load_dataset
import torch
import sentencepiece as spm
import os
tokenizer = spm.SentencePieceProcessor(model_file='/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model')
from tqdm import tqdm
import pandas as pd

TASK = '/paws-x'
LANG = 'en'
CONTEXT_LEN = 256

# data = pd.read_csv(f'glue{TASK}/train.tsv',sep='\t',header=None)
# print(data.columns)
# print(data[:2])
INP_INDEX_1 = 'sentence1'
INP_INDEX_2 = 'sentence2'
OUTP_INDEX = 'label'
data = load_dataset('paws-x',name=LANG)

if not os.path.exists(f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/')

if not os.path.exists(f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/')

train_1 = []
train_2 = []
train_mask = []


for row in data['train']:
    inp_1 = row[INP_INDEX_1]
    inp_2 = row[INP_INDEX_2]
    outp = row[OUTP_INDEX]
    train_1.append(inp_1)
    train_2.append(inp_2)
    train_mask.append(outp)


train_1 = tokenizer.encode(train_1)
train_2 = tokenizer.encode(train_2)

temp = []

for i in tqdm(range(len(train_1))):
    temp.append([1] + train_1[i] + [2] + train_2[i] + [2])

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





train_1 = []
train_2 = []
train_mask = []


for row in data['test']:
    inp_1 = row[INP_INDEX_1]
    inp_2 = row[INP_INDEX_2]
    outp = row[OUTP_INDEX]
    train_1.append(inp_1)
    train_2.append(inp_2)
    train_mask.append(outp)


train_1 = tokenizer.encode(train_1)
train_2 = tokenizer.encode(train_2)

temp = []

for i in tqdm(range(len(train_1))):
    temp.append([1] + train_1[i] + [2] + train_2[i] + [2])

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

