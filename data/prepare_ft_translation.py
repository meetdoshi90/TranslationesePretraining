from datasets import load_dataset
import torch
import sentencepiece as spm
import os
tokenizer = spm.SentencePieceProcessor(model_file='/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model')
from tqdm import tqdm
import pandas as pd


TASK = 'ai4bharat/samanantar'
SRC_LANG = 'en'
TGT_LANG = 'hi'
CONTEXT_LEN = 1024
REVERSE = True
SIZE = 1000000

data = load_dataset(TASK,name=TGT_LANG, streaming=True)
shuffled_dataset = data.shuffle(seed=42, buffer_size=1000)

if not os.path.exists('samanantar'):
    os.mkdir('samanantar')

train = []
train_mask = []
val = []
val_mask = []

count = 0
pbar = tqdm(total=SIZE)
for row in tqdm(shuffled_dataset['train']):
    #print(row)
    train.append(row['src'])
    train_mask.append(row['tgt'])
    pbar.update(1)
    count+=1
    if count >= SIZE:
        break
    #break
print(len(train),len(train_mask))

if REVERSE:
    temp = train
    train = train_mask
    train_mask = temp
    temp = SRC_LANG
    SRC_LANG = TGT_LANG
    TGT_LANG = temp

df = pd.DataFrame({
    'src': train,
    'tgt': train_mask
})

df.to_csv(f'samanantar/{SRC_LANG}-{TGT_LANG}.csv',index=False)

train = tokenizer.encode(train)
train_mask = tokenizer.encode(train_mask)

temp = []
temp_1 = []

for i in tqdm(range(len(train))):
    temp.append([1] + train[i] + [2] + train_mask[i] + [2])
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


if not os.path.exists(f'./train-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./train-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/')

if not os.path.exists(f'./val-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/'):
    os.mkdir(f'./val-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/')


torch.save(train,f'./train-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/train.pt')
torch.save(train_mask,f'./train-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/train.pt.mask')

i=0
for j in tqdm(range(0,train.shape[0],2000)):
    i+=1
    t = train[j:j+2000,:].clone()
    tm = train_mask[j:j+2000,:].clone()
    print(t.shape,tm.shape)
    torch.save(t,f'./val-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/val-{i}.pt')
    torch.save(tm,f'./val-data-ft-{SRC_LANG}-{TGT_LANG}-{TASK.split("/")[1]}/val-{i}.pt.mask')
    if i==3:
        break










