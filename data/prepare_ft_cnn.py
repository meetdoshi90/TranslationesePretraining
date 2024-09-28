from datasets import load_dataset
import torch
import sentencepiece as spm
import os
tokenizer = spm.SentencePieceProcessor(model_file='/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model')
from tqdm import tqdm


TASK = 'knkarthick/dialogsum'
DATASET = 'knkarthick/dialogsum'
LANG = 'english'
CONTEXT_LEN = 1024
BATCH_SIZE = 500

SRC = 'dialogue'
TGT = 'summary'


data = load_dataset(DATASET)


if not os.path.exists(f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/')

if not os.path.exists(f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/')


train = []
train_mask = []
val = []
val_mask = []


for row in data['train']:
    #print(row)
    train.append(row[SRC])
    train_mask.append(row[TGT])
    #break


train = tokenizer.encode(train)
train_mask = tokenizer.encode(train_mask)

temp = []
temp_1 = []

for i in tqdm(range(len(train))):
    formatted_inp = [1] + train[i] + [2] + train_mask[i] + [2]
    if len(formatted_inp)>(CONTEXT_LEN+1):
        continue
    temp.append(formatted_inp)
    temp_1.append([0]*len(train[i]) + [0] + [1]*len(train_mask[i]) + [1])

train = temp
train_mask = temp_1

clip = 0
total = 0
for i in tqdm(range(len(train))):
    total +=1 
    if len(train[i])>(CONTEXT_LEN+1):
        clip+=1
        train[i] = train[i][:CONTEXT_LEN+1]
    else:
        train[i] = train[i] + [3]*(CONTEXT_LEN+1-len(train[i]))

    if len(train_mask[i])>(CONTEXT_LEN):
        train_mask[i] = train_mask[i][:CONTEXT_LEN]
    else:
        train_mask[i] = train_mask[i] + [3]*(CONTEXT_LEN-len(train_mask[i]))

    assert len(train_mask[i])==CONTEXT_LEN
    assert len(train[i])==(CONTEXT_LEN+1)

train = torch.IntTensor(train)
train_mask = torch.IntTensor(train_mask)
print('CLIP', clip, total)
print(train.shape, train_mask.shape)

torch.save(train,f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/train.pt')
torch.save(train_mask,f'./train-data-ft-{LANG}-{TASK.split("/")[1]}/train.pt.mask')

train = []
train_mask = []
val = []
val_mask = []


for row in data['validation']:
    #print(row)
    train.append(row[SRC])
    train_mask.append(row[TGT])
    #break


train = tokenizer.encode(train)
train_mask = tokenizer.encode(train_mask)

temp = []
temp_1 = []

for i in tqdm(range(len(train))):
    formatted_inp = [1] + train[i] + [2] + train_mask[i] + [2]
    if len(formatted_inp)>(CONTEXT_LEN+1):
        continue
    temp.append(formatted_inp)
    temp_1.append([0]*len(train[i]) + [0] + [1]*len(train_mask[i]) + [1])

train = temp
train_mask = temp_1

clip = 0
total = 0
for i in tqdm(range(len(train))):
    total += 1
    if len(train[i])>(CONTEXT_LEN+1):
        clip+=1
        train[i] = train[i][:CONTEXT_LEN+1]
    else:
        train[i] = train[i] + [3]*(CONTEXT_LEN+1-len(train[i]))

    if len(train_mask[i])>(CONTEXT_LEN):
        train_mask[i] = train_mask[i][:CONTEXT_LEN]
    else:
        train_mask[i] = train_mask[i] + [3]*(CONTEXT_LEN-len(train_mask[i]))

    assert len(train_mask[i])==CONTEXT_LEN
    assert len(train[i])==(CONTEXT_LEN+1)

train = torch.IntTensor(train)
train_mask = torch.IntTensor(train_mask)
print('CLIP', clip, total)

i=0
for j in tqdm(range(0,train.shape[0],BATCH_SIZE)):
    i+=1
    t = train[j:j+BATCH_SIZE,:].clone()
    tm = train_mask[j:j+BATCH_SIZE,:].clone()
    print(t.shape,tm.shape)
    torch.save(t,f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/val-{i}.pt')
    torch.save(tm,f'./val-data-ft-{LANG}-{TASK.split("/")[1]}/val-{i}.pt.mask')
    if i==4:
        break










