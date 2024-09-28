import torch
print(torch.tensor([1,2,3]))
import sentencepiece as spm
import os
tokenizer = spm.SentencePieceProcessor(model_file='/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model')
from tqdm import tqdm

SRC_DIR = './hi/'
CONTEXT_LEN=1024
token_limit = 1190112000
#token_limit = 2387112000
PARALLEL = False


import random

files = os.listdir(SRC_DIR)

if PARALLEL==True:
    if SRC_DIR=='./syn-GU/':
        files = [f'{i}.txt.guj_Gujr.merged' for i in range(1,26)]
    elif SRC_DIR=='./syn-HI/':
        files = [f'{i}.txt.hin_Deva.merged' for i in range(1,26)]
    elif SRC_DIR=='./EN/':
        files = [f'OSCAR_2301_en_meta_part_{i}.txt' for i in range(1,26)]
    elif SRC_DIR == './syn-EN-hi/':
        files = [f'{i}.txt.himC4-eng_Latn.merged' for i in range(1,21)]
else:
    if SRC_DIR=='./syn-GU/':
        files = [f'{i}.txt.guj_Gujr.merged' for i in range(1,26)]
        files.reverse()
    elif SRC_DIR=='./syn-HI/':
        files = [f'{i}.txt.hin_Deva.merged' for i in range(1,26)]
        files.reverse()
    elif SRC_DIR=='./EN/':
        files = [f'OSCAR_2301_en_meta_part_{i}.txt' for i in range(1,26)]
    elif SRC_DIR == './syn-EN-hi/':
        files = [f'{i}.txt.himC4-eng_Latn.merged' for i in range(1,21)]



print(files)
def splitter(li):
    li = [li]
    splitted_li = []
    for l in tqdm(li):
        chunks = [l[x:x+CONTEXT_LEN+1] for x in range(0, len(l), CONTEXT_LEN)]
        for i in range(len(chunks)):
            if len(chunks[i])!=(CONTEXT_LEN+1):
                chunks[i].extend([3]*(CONTEXT_LEN+1-len(chunks[i])))
        splitted_li.extend(chunks)
    return splitted_li



tc = 0
for file in files:
    # if '2301' not in file:
    #     continue
    # if int(file.split('_')[-1].split('.')[0])<=10:
    #     continue
    if not os.path.exists(SRC_DIR+file):
        print('File', file,' Does not exist')
        continue
    print(file)
    with open(SRC_DIR+file) as f:
        data = f.readlines()
    for i in tqdm(range(len(data))):
        data[i] = data[i].strip()
        if data[i] == '< DOC _ START >' or data[i]=='<DOC _ START>':
            data[i] = '<DOC_START>'
        elif data[i] == '< DOC _ END >' or data[i]=='<DOC _ END>':
            data[i] = '<DOC_END>'

    limit_data = []
    for i in tqdm(range(len(data))):
        if tc > token_limit:
            print('Total tokens included = ', tc)
            break
        limit_data.append(data[i])
        tc += len(data[i].split())
    if len(limit_data)==0:
        break
    data = limit_data
    #print(data[:10])
    #random.shuffle(data)
    #data = sorted(data, key=len)
    tokenized_data = tokenizer.encode(data)
    #tokenized_data = tokenizer.decode(tokenized_data)
    #print(tokenized_data[:10])
    #print(tokenizer.decode(tokenized_data[:10]))
    #print(tokenizer.encode_as_pieces(data[:10]))
    # for i in range(100):
    #     print(tokenizer.decode(tokenized_data[i]))
    #     print(" ".join(tokenizer.encode_as_pieces(data[i])))
    #     print(tokenizer.encode(data[i]))
    #     print(data[i])
    #     print('#'*30)
    data = []
    tokenized_file = []
    temp = []
    token_count = 0
    doc_count = 0
    for row in tqdm(tokenized_data):
        if row == [48554, 48860, 48682, 13096, 48766, 2806, 21873, 48808] or row ==[48554, 48860, 48682, 13096, 48766, 46319, 48808]:
            if temp!=[]:
                tokenized_file.extend([1] + temp + [2])
                doc_count += 1
            temp = []
        else:
            temp.extend(row)
            token_count += len(row)
    if temp!=[]:
        tokenized_file.extend(temp)
        doc_count+=1
    temp = []
    tokenized_data=[]
    tokenized_file = splitter(tokenized_file)
    
    # for row in tqdm(tokenized_data):
    #     tokenized_output = [1] + row + [2]
    #     if len(tokenized_output)>CONTEXT_LEN+1:
    #         chunks = [tokenized_output[x:x+CONTEXT_LEN+1] for x in range(0, len(tokenized_output), CONTEXT_LEN)]
    #         for i in range(len(chunks)):
    #             if len(chunks[i])!=(CONTEXT_LEN+1):
    #                 chunks[i].extend([3]*(CONTEXT_LEN+1-len(chunks[i])))
    #         tokenized_file.extend(chunks)
    #     elif len(tokenized_output)<CONTEXT_LEN+1:
    #         tokenized_output.extend([3]*(CONTEXT_LEN+1-len(tokenized_output)))
    #         tokenized_file.append(tokenized_output)
    #     else:
    #         tokenized_file.append(tokenized_output)

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
    print('Doc count', doc_count)
    print('Token count', token_count)
    print((data==3).sum().item()/torch.numel(data))
    print((data==0).sum().item()/torch.numel(data))
    torch.save(data,f'./torch_data/{file}.pt')
    
    # with open(f'./torch_data/{file}.txt','w') as f:
    #     tokenized_file = [str(x) for x in tokenized_file]
    #     f.writelines(' '.join(tokenized_file))
    #break

print('Total tokens included = ', tc)