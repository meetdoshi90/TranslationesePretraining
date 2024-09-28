from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import sentencepiece as spm
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
import sys
from huggingface_hub import notebook_login,login

login(token='INSERT_TOKEN_HERE')

if not os.path.exists('./scores/'):
    os.mkdir('./scores/')

tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
SYN = str(sys.argv[1]) # ''
SYN = '' if SYN=='-' else SYN 
TASK = str(sys.argv[2]) #'IndicHeadlineGeneration'
LANG = str(sys.argv[3]) #'hi'
SIZE = str(sys.argv[4]) #'small'
BATCH_SIZE = 1
MAX_COUNT = 12000
NUM_BEAMS = 5
LEN_PENALTY = 1.0
NO_N_GRAM_REPEAT = 4
MAX_NEW_TOKENS=64
EARLY_STOPPING=True

dataset_name = f'{TASK.split("-")[0]}'
if 'dialogsum' in dataset_name:
    dataset = load_dataset(dataset_name, split='test')
else:
    dataset = load_dataset(dataset_name, name=LANG, split='test')

for d in dataset:
    print(d.keys())
    

model = AutoModelForCausalLM.from_pretrained(f'meetdoshi90/MiniLM-{SYN}-{SIZE}-{TASK.split("/")[1]}',trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(f'meetdoshi90/MiniLM-BI-{SYN}-{SIZE}-{TASK}',trust_remote_code=True)
model.eval()
#model.half()
model.to('cuda')

inp = []
tgt = []
gens = []

count = 0
for row in tqdm(dataset):
    if TASK=='ccdv/cnn_dailymail':
        inp.append(row['article'])
        tgt.append(row['highlights'])
    elif TASK=='csebuetnlp/xlsum-headline':
        inp.append(row['text'])
        tgt.append(row['title'])
    elif TASK=='csebuetnlp/xlsum-summarization':
        inp.append(row['text'])
        tgt.append(row['summary'])
    elif TASK=='knkarthick/dialogsum':
        inp.append(row['dialogue'])
        tgt.append(row['summary'])
    count+=1
    if count>=MAX_COUNT:
        break

enc_text = tokenizer.encode(inp)
temp = tokenizer.encode(tgt)
maxl = 0
for t in temp:
    maxl = max(maxl,len(t))
print(maxl)
print(len(enc_text))
#print(enc_text)

for i in tqdm(range(len(inp))):
    enc_inp = torch.tensor([1] + enc_text[i] + [2]).view(1,-1).to('cuda')
    #print(enc_inp.shape)
    with torch.no_grad():
        out = model.generate(
            enc_inp,
            num_beams=NUM_BEAMS,
            length_penalty=LEN_PENALTY,
            max_new_tokens=MAX_NEW_TOKENS,
            no_repeat_ngram_size=NO_N_GRAM_REPEAT,
            early_stopping=EARLY_STOPPING,
            eos_token_id=[2]
        )
    out = out[0].tolist()[enc_inp.shape[1]:]
    dec_out = tokenizer.decode(out)
    print(tgt[i])
    print('-'*30)
    print(dec_out)
    gens.append(dec_out)

df = pd.DataFrame({
    'target': tgt,
    'generated': gens,
    'input': inp,
})

df.to_csv(f'./scores/MiniLM-{SYN}-{SIZE}-{TASK.split("/")[1]}.csv',index=False)




