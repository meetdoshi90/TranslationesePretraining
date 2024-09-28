import torch
import torch.nn as nn
import sentencepiece as spm
from flores import flores_codes, iso_to_flores
torch.backends.cuda.enable_flash_sdp(True) #Enable flash scaled dot product attention
torch.backends.cuda.enable_mem_efficient_sdp(True) #Enable mem efficient SDP
torch.backends.cuda.enable_math_sdp(True) #Math sdp
from torch.utils.data import Dataset, DataLoader
#Print status
import os
print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
from models.gpt2_rope_inf import GPTModel
import lightning.pytorch as pl
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate import unicode_transliterate
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from indicnlp.tokenize import indic_detokenize
from sacremoses import MosesDetokenizer
from normalise_punctuation import punc_norm
from tqdm import tqdm
import json
from time import time,sleep
#exit()
model_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints-en-tiny_4096/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'
def nl(input, target): return -input[range(target.shape[0]), target].log().mean()

CONTEXT_LEN=4096
BATCH_SIZE=16

class TokenizedDataset(Dataset):
    def __init__(self, text_file):
        text_file = text_file.to('cuda')
        self.text_file = text_file
        self.file_size_ = self.text_file.shape[0]
        print('length of text file', self.text_file.shape)
        
    def __getitem__(self, idx):
        tokenized_output = self.text_file[idx,:]
        return tokenized_output

    def __len__(self):
        return self.file_size_


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

def perplexity(model,tokenizer,enc_text):
    #model.eval()
    ppl_scores = None
    #loss = torch.nn.CrossEntropyLoss(reduction='none')
    '''
    for i in range(1,enc_text.shape[1]):
        prompt = enc_text[:,:i]
        #print(prompt.shape)
        #print(tokenizer.decode(prompt.view(-1,1).tolist()))
        logits,hidden_states = model(prompt)
        logits = logits[:, -1, :]
        logit_probs = torch.nn.functional.softmax(logits, dim=-1)
        target_probs = torch.zeros_like(logit_probs)
        target_probs[:,enc_text[:,i]] = 1
        #print(logit_probs.shape, logit_probs.sum())
        #print(target_probs.shape, target_probs.sum())
        if ppl_scores==None:
            ppl_scores = loss(logits,enc_text[:,i]).view(1,-1)
        else:
            ppl_scores = torch.cat([ppl_scores,loss(logits,enc_text[:,i]).view(1,-1)])
        #print(hidden_states[:,-1,:].shape)
        if hidden==None:
            hidden = hidden_states[:,-1,:]
        else:
            hidden = torch.cat([hidden,hidden_states[:,-1,:]])
        #print(nl(logit_probs,enc_text[:,i]))
    print(ppl_scores)
    '''
    #print(enc_text.shape)
    inps = enc_text[:,:-1]
    tgts = enc_text[:,1:]
    B,C = inps.shape 
    V = 56000
    st = time()
    with torch.no_grad():
        logits,_ = model(inps)
    print('Time for forward', time()-st)
    #print(logits[0,:-1,:].shape,hidden_states[0,:-1,:].shape,enc_text[0,1:].shape)
    st = time()
    ppl_scores = nn.functional.cross_entropy(logits.contiguous().view(B*C,V),tgts.contiguous().view(B*C),reduction='none')
    #hidden_states = hidden_states.to('cpu')
    #ppl_scores = ppl_scores.to('cpu')
    print('Misc time',time()-st)
    #print(hidden_states.get_device(),ppl_scores.get_device())
    #print(hidden_states[0,:-1,:].shape, ppl_scores.shape)
    #del logits
    return ppl_scores.view(B,C)


model = GPTModel.load_from_checkpoint(model_path)
model.eval()
model.cuda()

for n,p in model.named_parameters():
    p.requires_grad = False
    print(n,p.requires_grad)

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
print('Loaded models...')

data = []


SKIP_K = 10

FOLDER = './data/syn-EN-gu/'
FOLDER_FILES = reversed(sorted(os.listdir(FOLDER)))[:1]
print('Folder files', FOLDER_FILES)
if not os.path.exists(FOLDER[:-1]+'-scores/'):
    os.mkdir(FOLDER[:-1]+'-scores/')

for file in tqdm(FOLDER_FILES):
    final_scores = []
    with open(FOLDER+file,'r+', encoding="utf-8") as f:
        data = f.readlines()
        data= data[:100000]
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
    dataset = TokenizedDataset(tokenized_file)
    data = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    torch.cuda.synchronize()
    print(type(data))
    #tokenized_file = tokenized_file.contiguous().to('cuda')
    #print('Loaded to cuda')
    # temp = []
    # for i in range(0,len(tokenized_file),BATCH_SIZE):
    #     temp.append(tokenized_file[i:i+BATCH_SIZE,:])
    # tokenized_file = temp

    #sleep(100)
    #exit()

    # temparr = []
    # for i in tqdm(range(0,len(tokenized_file),BATCH_SIZE)):
    #     batch_enc_inp = None
    #     for j in range(i,min(i+BATCH_SIZE,len(tokenized_file))):
    #         temp = tokenized_file[i] + [3]*max(0,CONTEXT_LEN+1-len(tokenized_file[i]))
    #         temp = temp[:CONTEXT_LEN+1]
    #         #tokenized_file[i+j] = temp
    #         assert len(temp)==CONTEXT_LEN+1
    #         enc_inp = torch.tensor(temp).view(1,-1).long()
    #         if batch_enc_inp==None:
    #             batch_enc_inp = enc_inp
    #         else:
    #             batch_enc_inp = torch.cat([batch_enc_inp,enc_inp],dim=0)
    #     #print(batch_enc_inp.shape)
    #     batch_enc_inp = batch_enc_inp.to('cuda')
    #     temparr.append(batch_enc_inp)
    # tokenized_file = temparr


    ok = time()
    print('STARTING SCORING')
    
    #for i in tqdm(range(0,len(tokenized_file),BATCH_SIZE)):
    for batch_enc_inp in data:
        stt = time()
        #batch_enc_inp = batch_ids.clone()
        #torch.cuda.empty_cache()
        #torch.cuda.synchronize()
        print(batch_enc_inp.shape)
        #batch_enc_inp = tokenized_file[i:i+BATCH_SIZE,:]
        # batch_enc_inp = None
        # cur_batch = 0
        # st = time()
        # for j in range(i,min(i+BATCH_SIZE,len(tokenized_file))):
        #     cur_batch+=1
        #     temp = tokenized_file[i] + [3]*max(0,CONTEXT_LEN+1-len(tokenized_file[i]))
        #     temp = temp[:CONTEXT_LEN+1]
        #     #tokenized_file[i+j] = temp
        #     assert len(temp)==CONTEXT_LEN+1
        #     enc_inp = torch.tensor(temp).view(1,-1).long()
        #     if batch_enc_inp==None:
        #         batch_enc_inp = enc_inp
        #     else:
        #         batch_enc_inp = torch.cat([batch_enc_inp,enc_inp],dim=0)
        # print('For loop time',time()-st)
        st = time()
        assert batch_enc_inp.shape[1]==CONTEXT_LEN+1
        print('Assert time', time()-st)
        st = time()
        #torch.cuda.synchronize()
        batch_enc_inp = batch_enc_inp.to('cuda')
        print('Cuda time' ,time()-st)
        #print(enc_inp.shape)
        ppl = perplexity(model,tokenizer,batch_enc_inp)
        #hid = hid.cpu()
        st = time()
        #ppl = ppl.cpu()
        #print(hid.shape,ppl.shape)
        #enc_inp = enc_inp.cpu()
        #batch_enc_inp = batch_enc_inp.cpu()
        #final_states.append(torch.flatten(hid.detach(),end_dim=-2))
        # for j in range(ppl.shape[0]):
        #     seq_score = ppl[j,:].tolist()
        #     final_scores.append(seq_score)
        final_scores.extend(ppl.detach())
        print('Time for other',time()-st)
        print('Time for single loop',time()-ok)
        #print([i.get_device() for i in final_scores])
        #print([i.get_device() for i in final_states])
    #ppl_scores = final_scores
    #hidden_states = torch.cat(final_states)
    print(len(final_scores))
    assert len(final_scores)==len(tokenized_file)
    #print(hidden_states.shape)
    out = []
    for i in tqdm(range(len(final_scores))):
        out.append(
            {
                'id': i,
                'scores': final_scores[i],
                'tokens': tokenized_file[i]
            }
        )
    exit()
    with open(FOLDER[:-1]+'-scores/'+file+'.json','w') as f:
        json.dump(out,f)




