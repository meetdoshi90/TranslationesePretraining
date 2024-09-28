from datasets import load_dataset
import torch 
from mosestokenizer import *
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from torchmetrics.classification import BinaryMatthewsCorrCoef, F1Score, BinaryF1Score, MulticlassMatthewsCorrCoef
matthews_corr = BinaryMatthewsCorrCoef()
from tqdm import tqdm
import random
# data = load_dataset('glue',name='cola')

#pipe = pipeline("text-classification", model="textattack/roberta-base-CoLA",device=0)

# preds = []
# tgts = []
# for t in tqdm(data['validation']):
#     out = pipe(t['sentence'])[0]
#     if out['label'] == 'LABEL_0':
#         preds.append(0)
#     elif out['label'] == 'LABEL_1':
#         preds.append(1)
#     else:
#         print(out)
#     tgts.append(t['label'])

# print(preds)
# print(tgts)
# print(f'Acc = {(torch.tensor(preds)==torch.tensor(tgts)).sum()/len(preds)}')
# print(matthews_corr(torch.tensor(preds),torch.tensor(tgts)))

# with MosesSentenceSplitter('en') as splitsents:
#     splitted_sentences = splitsents([
#         'Mr. Smith is away.  What should I do? I do not have much time'
#         ])
    
# print(splitted_sentences)

tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
model.eval()
model.to('cuda')
#with open('./data/EN/OSCAR_2301_en_meta_part_2.txt','r+', encoding="utf-8") as f:
with open('./data/syn-EN-gu/1.txt.gu22-eng_Latn.merged','r+', encoding="utf-8") as f:
#with open('./data/syn-EN-hi/2.txt.hi-eng_Latn.merged','r+', encoding="utf-8") as f:
    data = f.readlines()
    #data = data[:10000]
    #print(data[0])

for i in tqdm(range(len(data))):
    data[i] = data[i].strip()
    if data[i] == '< DOC _ START >' or data[i]=='<DOC _ START>':
        data[i] = '<DOC_START>'
    elif data[i] == '< DOC _ END >' or data[i]=='<DOC _ END>':
        data[i] = '<DOC_END>'

doc = '\n'.join(data)

doc = doc.replace('<DOC_START>\n','')

doc = doc.split('<DOC_END>')

print(len(doc))

random.shuffle(doc)

doc = doc[:1000]

splitted_sentences = []

for d in tqdm(doc):
    with MosesSentenceSplitter('en') as splitsents:
        splitted_sentences.extend(splitsents([
            d
            ]))
        
print('Total sentences = ', len(splitted_sentences))

preds = []
for t in tqdm(splitted_sentences):
    tokenized_input = tokenizer.encode(t)[:512]
    tokenized_input = torch.tensor(tokenized_input).view(1,-1).to('cuda')
    out = torch.argmax(model(tokenized_input).logits[0]).detach().item()
    #print(out)
    #break
    preds.append(out)

print('Percentage of sentences labelled as acceptable = ', sum(preds)/len(preds))