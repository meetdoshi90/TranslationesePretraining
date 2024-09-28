import torch
print(torch.tensor([1,2,3]))
import sentencepiece as spm
import os
tokenizer = spm.SentencePieceProcessor(model_file='/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/Extra_Merged_SS.model')
from tqdm import tqdm

SRC_DIR = './train-data/'

CONTEXT_LEN=96
import random

files = os.listdir(SRC_DIR)

for file in files:
    print(file)
    with open(SRC_DIR+file) as f:
        data = f.readlines()
    #random.shuffle(data)
    data = sorted(data, key=len)
    tokenized_data = tokenizer.encode(data)
    data = []
    tokenized_file = []
    for row in tqdm(tokenized_data):
        tokenized_output = [1] + row + [2]
        if len(tokenized_output)>CONTEXT_LEN+1:
            chunks = [tokenized_output[x:x+CONTEXT_LEN+1] for x in range(0, len(tokenized_output), CONTEXT_LEN)]
            for i in range(len(chunks)):
                if len(chunks[i])!=(CONTEXT_LEN+1):
                    chunks[i].extend([3]*(CONTEXT_LEN+1-len(chunks[i])))
            tokenized_file.extend(chunks)
        elif len(tokenized_output)<CONTEXT_LEN+1:
            tokenized_output.extend([3]*(CONTEXT_LEN+1-len(tokenized_output)))
            tokenized_file.append(tokenized_output)
        else:
            tokenized_file.append(tokenized_output)

    #verify
    data = torch.IntTensor(tokenized_file)
    # for i in tqdm(range(len(tokenized_file))):
    #     if len(tokenized_file[i])!=(CONTEXT_LEN+1):
    #         print(tokenized_file[i])
    #         print(len(tokenized_file[i]))
    #         print('Mismatch')
    #         exit()
    #     else:
    #         if data == []:
    #             # print(tokenized_file[i])
    #             data = torch.tensor(tokenized_file[i]).view(1,-1)
    #         else:
    #             # print(tokenized_file[i])
    #             # print(data.shape, torch.tensor(tokenized_file[i]).shape)
    #             #data = torch.vstack((data,torch.tensor(tokenized_file[i]).view(1,-1)))
    #             data = torch.cat((data,torch.tensor(tokenized_file[i]).view(1,-1)),0)
    
    print(data.shape)
    print((data==3).sum().item()/torch.numel(data))
    torch.save(data,f'{file}.pt')
