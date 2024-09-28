import torch
import sentencepiece as spm
from flores import flores_codes, iso_to_flores
torch.backends.cuda.enable_flash_sdp(True) #Enable flash scaled dot product attention
torch.backends.cuda.enable_mem_efficient_sdp(True) #Enable mem efficient SDP
torch.backends.cuda.enable_math_sdp(True) #Math sdp
#Print status
print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
from models.gpt2_rope_inf import GPTModel
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
#exit()
model_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints-en-small/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'
def nl(input, target): return -input[range(target.shape[0]), target].log().mean()

CONTEXT_LEN=1024
#BATCH_SIZE=64

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
    model.eval()
    ppl_scores = None
    hidden_states = None
    hidden = None
    loss = torch.nn.CrossEntropyLoss(reduction='none')
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
    with torch.no_grad():
        logits, hidden_states = model(enc_text)
    #print(logits[0,:-1,:].shape,hidden_states[0,:-1,:].shape,enc_text[0,1:].shape)
    ppl_scores = loss(logits[:,:-1,:].to('cpu').permute(0,2,1),enc_text[:,1:].to(torch.int64).to('cpu'))
    #hidden_states = hidden_states.to('cpu')
    ppl_scores = ppl_scores.to('cpu')
    #print(hidden_states.get_device(),ppl_scores.get_device())
    #print(hidden_states[0,:-1,:].shape, ppl_scores.shape)
    del logits
    return ppl_scores


model = GPTModel.load_from_checkpoint(model_path)
model.eval()

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
print('Loaded models...')

class Config:
    context_len = 1024
config = Config()

#text = "My name is Meet."
#enc_inp = [1] + tokenizer.encode(text)
#enc_inp = torch.tensor(enc_inp).view(1,-1).to('cuda')
#print(text)
#print(enc_inp.dtype)
#print(tokenizer.decode(enc_inp.view(-1,1).tolist()))

#hidden_states, ppl_scores = perplexity(model,tokenizer, enc_inp)
#print(ppl_scores)
#exit()
data = []

#final_states = []
final_scores = []

with open('./data/EN/OSCAR_2301_en_meta_part_26.txt','r+', encoding="utf-8") as f:
#with open('./data/syn-EN-gu/1.txt.gu23-eng_Latn.merged','r+', encoding="utf-8") as f:
#with open('./data/syn-EN-hi/15.txt.hi-eng_Latn.merged','r+', encoding="utf-8") as f:
    data = f.readlines()
    data = data[:10000]
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
temp = []
#tokenized_file = splitter(tokenized_file)
print('Doc count =', doc_count)
#data = torch.IntTensor(tokenized_file)
#print(data.shape)

print('STARTING SCORING')
for i in tqdm(range(len(tokenized_file))):
    enc_inp = torch.tensor(tokenized_file[i]).to('cuda').view(1,-1).long()
    #print(enc_inp.shape)
    ppl = perplexity(model,tokenizer,enc_inp)
    #hid = hid.cpu()
    ppl = ppl.cpu()
    #print(hid.shape,ppl.shape)
    enc_inp = enc_inp.cpu()
    #final_states.append(torch.flatten(hid.detach(),end_dim=-2))
    final_scores.append(torch.flatten(ppl.detach()))

    #print([i.get_device() for i in final_scores])
    #print([i.get_device() for i in final_states])
#ppl_scores = final_scores
#hidden_states = torch.cat(final_states)
print(len(final_scores))
#print(hidden_states.shape)
out = []
for i in tqdm(range(len(tokenized_file))):
    out.append(
        {
            'tokens': tokenized_file[i],
            'scores': final_scores[i].tolist()
        }
    )

with open('./data/ppl/scores-en-orig.json','w') as f:
    json.dump(out,f)

#hidden_states = hidden_states

#print(hidden_states.shape)
#temp = ppl_scores.tolist()
#print(len([i for i in temp if i>1.0]), len(ppl_scores))
#torch.save(ppl_scores,'./data/ppl/scores-en-orig.pt')
#torch.save(hidden_states,'./data/ppl/activations.pt')



