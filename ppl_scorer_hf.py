from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import sentencepiece as spm
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
import sys
from time import time,sleep
from huggingface_hub import notebook_login,login

login(token='INSERT_TOKEN_HERE')


tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(f'meetdoshi90/MiniLM-base-en-tiny_4096',trust_remote_code=True)
model = model.eval()
model = model.to('cuda')



SKIP_K = 10
CONTEXT_LEN = 4096
BATCH_SIZE = 8

FOLDER = './data/syn-EN-gu/'
FOLDER_FILES = reversed(sorted(os.listdir(FOLDER)))
FOLDER_FILES = ['1.txt.gu23-eng_Latn.merged']
print('Folder files', FOLDER_FILES)
if not os.path.exists(FOLDER[:-1]+'-scores/'):
    os.mkdir(FOLDER[:-1]+'-scores/')

for file in tqdm(FOLDER_FILES):
    print(file)
    final_scores = []
    with open(FOLDER+file,'r+', encoding="utf-8") as f:
        data = f.readlines()
        print(data[0])

    for i in tqdm(range(len(data))):
        data[i] = data[i].strip()
        if data[i] == '< DOC _ START >' or data[i]=='<DOC _ START>':
            data[i] = '<DOC_START>'
        elif data[i] == '< DOC _ END >' or data[i]=='<DOC _ END>':
            data[i] = '<DOC_END>'

    limit_data = []
    for i in tqdm(range(len(data))):
        limit_data.append(data[i])
    data = limit_data
    limit_data = []

    tokenized_data = tokenizer.encode(data)

    data = []
    tokenized_file = []
    temp = []
    token_count = 0
    doc_count = 0
    for row in tqdm(tokenized_data):
        if row == [48554, 48860, 48682, 13096, 48766, 2806, 21873, 48808] or row ==[48554, 48860, 48682, 13096, 48766, 46319, 48808]:
            if temp!=[]:
                tokenized_file.extend([[1] + temp + [2]])
                doc_count += 1
            temp = []
        else:
            temp.extend(row)
            token_count += len(row)
    if temp!=[]:
        tokenized_file.extend([temp])
        doc_count+=1
    #tokenized_file = splitter(tokenized_file)
    print('Doc count =', doc_count)
    #data = torch.IntTensor(tokenized_file)
    #print(data.shape)


    temp = []
    for row in tqdm(tokenized_file):
        doc = row + [3]*max(0,CONTEXT_LEN+1-len(row))
        doc = doc[:CONTEXT_LEN+1]
        assert len(doc) == CONTEXT_LEN+1
        temp.append(doc)
    tokenized_file = torch.tensor(temp).long()
    print(tokenized_file.shape)
    temp = []
    loss = []

    for i in tqdm(range(0,tokenized_file.shape[0],BATCH_SIZE)):
        batch_inp = tokenized_file[i:i+BATCH_SIZE,:].to('cuda')
        out = model(batch_inp,labels=batch_inp,return_dict=True)
        loss.extend(out.loss.tolist())

    loss = torch.tensor(loss)
    print('loss shape',loss.shape)

    save_name_tokens = FOLDER[:-1]+'-scores/'+file+'.tokens.pt'
    save_name_loss = FOLDER[:-1]+'-scores/'+file+'.loss.pt'
    torch.save(tokenized_file,save_name_tokens)
    torch.save(loss, save_name_loss)
    print(f'{file} done')
    tokenized_file=None 
    loss = None 
    tokenized_data = None

    del tokenized_data,tokenized_file,loss
    torch.cuda.empty_cache()
    break
