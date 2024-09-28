import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
torch.backends.cuda.enable_flash_sdp(True) #Enable flash scaled dot product attention
torch.backends.cuda.enable_mem_efficient_sdp(True) #Enable mem efficient SDP
torch.backends.cuda.enable_math_sdp(True) #Math sdp
#Print status
print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
from models.gpt2_rope_inf import GPTModel
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.regression import PearsonCorrCoef
import os
import sys 
VOCAB_SIZE = 56000
print(sys.argv)
NUM_CLASSES = int(sys.argv[3]) #1
BATCH_SIZE = int(sys.argv[4]) #48
EPOCHS = int(sys.argv[5]) #10
#PRINT_TRAIN_EVERY = 100
VAL_EVERY = 0.25
CONTEXT_LEN = 256
MODEL_NAME = str(sys.argv[1]) #'en-small'
TASK_NAME = str(sys.argv[2]) #'qnli'

print('\n'*5)
print(TASK_NAME,NUM_CLASSES,MODEL_NAME, BATCH_SIZE,EPOCHS, 'STARTING')
print('\n'*5)

# if NUM_CLASSES>1:
#     criterion = nn.CrossEntropyLoss()
# else:
#     criterion = nn.BCELoss()
criterion = nn.MSELoss()

# if NUM_CLASSES==1:
#     f1 = BinaryF1Score()
#     matthews_corr = BinaryMatthewsCorrCoef()
# else:
#     f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES)
#     matthews_corr = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES)

pearson = PearsonCorrCoef()

model_path = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints-{MODEL_NAME}/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'
checkpoint_dir = f'./checkpoints-ft-{MODEL_NAME}-{TASK_NAME}/'
TRAIN_DATAPATH = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/data/train-data-ft-en-{TASK_NAME}/train.pt'
VAL_DATAPATH = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/data/val-data-ft-en-{TASK_NAME}/val.pt'

model = GPTModel.load_from_checkpoint(model_path)
#model.eval()
model.train()
model.to('cuda')

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
print('Loaded models...')



class RegressionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, NUM_CLASSES, bias=False)
        self.act = nn.Sigmoid()
        torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        #x = self.act1(self.fc1(x))
        #x = self.act2(self.fc2(x))
        x = self.fc(x)
        if NUM_CLASSES==1:
            x = self.act(x)
        return x
    
regressionhead = RegressionHead()
regressionhead.to('cuda')

optimizer = optim.AdamW(list(model.parameters())+list(regressionhead.parameters()), lr=1e-5)


train_data = torch.load(TRAIN_DATAPATH)
val_data = torch.load(VAL_DATAPATH)
assert train_data.shape[1] == (CONTEXT_LEN+1)
assert val_data.shape[1] == (CONTEXT_LEN+1)
print(train_data[:,-1].sum(),train_data.shape[0])
print(train_data.shape,val_data.shape)
TRAIN_STEPS = train_data.shape[0]//BATCH_SIZE
VALSIZE = int(TRAIN_STEPS*VAL_EVERY)


def generate(model, prompt, max_tokens=30, temperature=1.0, config=None,num_beams=5, len_penalty=1.0, penalise_n_grams=4):
    """
    Generates text based on the provided prompt.
    Model determinism can be changed with temperature (range: [0, 1], higher means more unstable but creative predictions)
    """
    #model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            prompt = prompt
            #print(prompt)
            logits,x = model(prompt)
            #print(len(logits),logits[0].shape,logits[1].shape)
            logits = logits[:, -1, :] / temperature
            logit_probs = torch.nn.functional.softmax(logits, dim=-1)
            #next_prompt = torch.multinomial(logit_probs, num_samples=1)
            next_prompt = torch.topk(logit_probs, k=1, dim=-1).indices.view(-1,1)
            prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt

test = [1] + tokenizer.encode('My name is')
test = torch.tensor(test).view(1,-1).to('cuda')
test = generate(model,test)
print(test)
print(tokenizer.decode(test[0].tolist()))



def get_indexes(input_tensor,states):
    mask = input_tensor == 3
    result_tensor = torch.argmax(mask.int(), dim=1, keepdim=True)
    result_tensor = torch.where(mask.any(dim=1, keepdim=True), result_tensor, torch.tensor(0))
    result_tensor -= 1
    result_tensor = result_tensor.squeeze()
    #print(result_tensor)
    #print(input_tensor[torch.arange(input_tensor.shape[0]),result_tensor])
    out = states[torch.arange(input_tensor.shape[0]),result_tensor]
    assert out.shape == (states.shape[0],states.shape[2])
    return out

# test_1 = torch.tensor(
#     [
#         [0,3,3],
#         [4,5,6],
#         [7,8,3],
#         [10,1,12]
#     ]
# )

# test_2 = torch.arange(24).view(4,3,2)
# print(test_1.shape,test_2.shape)
# print('Get indexes')
# print(get_indexes(test_1,test_2))
# for n,p in classificationhead.named_parameters():
#     print(n,p)

labels = []
diff = []
predictions = []
tgts = []
for j in tqdm(range(0,val_data.shape[0],BATCH_SIZE)):
    with torch.no_grad():
        enc_text = val_data[j:j+BATCH_SIZE,:-1]
        enc_labels = val_data[j:j+BATCH_SIZE,-1]
        enc_text = enc_text.long()
        enc_text = enc_text.to('cuda')
        enc_labels = enc_labels.float()
        enc_labels = enc_labels.to('cuda')
        logits,states = model(enc_text)
        last_states = get_indexes(enc_text, states)
        #last_states = states[:,-1,:]
        classification_out = regressionhead(last_states)
        #print(classification_out)
        #probs = F.softmax(classification_out,dim=-1)
        #print(classification_out)
        if NUM_CLASSES>1:
            preds = torch.argmax(classification_out,dim=-1).squeeze()
        else:
            preds = (classification_out).squeeze()*5.0
        #print(preds.shape,enc_labels.shape)
        assert preds.shape == enc_labels.shape 
        diff.append(torch.mean(torch.abs(preds-enc_labels)))
        predictions.extend(preds.tolist())
        tgts.extend(enc_labels.tolist())
        labels.extend(preds.tolist())
print(f'Labels = {sum(labels)/len(labels)}')
print(f'Diff = {sum(diff)/len(diff)}')
print(f'Pearson = {pearson(torch.tensor(predictions)/5.0,torch.tensor(tgts)/5.0)}')

for epoch in tqdm(range(EPOCHS)):
    train_data = train_data[torch.randperm(train_data.size()[0])]
    for i in tqdm(range(0,train_data.shape[0],BATCH_SIZE)):
        optimizer.zero_grad()
        enc_text = train_data[i:i+BATCH_SIZE,:-1]
        enc_labels = train_data[i:i+BATCH_SIZE,-1]
        enc_text = enc_text.long()
        enc_text = enc_text.to('cuda')
        enc_labels = enc_labels.float()
        enc_labels = enc_labels.to('cuda')
        logits,states = model(enc_text)
        last_states = get_indexes(enc_text, states)
        #last_states = states[:,-1,:]
        #del enc_text, states,logits
        classification_out = regressionhead(last_states).squeeze()*5.0
        #print(classification_out.dtype,enc_labels.dtype)
        #print(last_states.shape)
        #print(last_states)
        #print(classification_out)
        #print(enc_labels)
        #classification_out = torch.round(classification_out)
        loss = criterion(classification_out,enc_labels)
        #old_grads = [p.grad for p in model.parameters()]
        loss.backward()
        #new_grads = [p.grad for p in model.parameters()]
        #print('\n'*10)
        #print(old_grads==new_grads)
        # for n,p in model.named_parameters():
        #     print(n,p.grad)
        optimizer.step()
        optimizer.zero_grad()
        #print(f' E: {epoch}, i: {i}, loss: {loss.item()}')

        if (i//BATCH_SIZE)%VALSIZE==0:
            print('Starting validation')
            predictions = []
            tgts = []
            labels = []
            diff = []
            for j in tqdm(range(0,val_data.shape[0],BATCH_SIZE)):
                with torch.no_grad():
                    enc_text = val_data[j:j+BATCH_SIZE,:-1]
                    enc_labels = val_data[j:j+BATCH_SIZE,-1]
                    enc_text = enc_text.long()
                    enc_text = enc_text.to('cuda')
                    enc_labels = enc_labels.float()
                    enc_labels = enc_labels.to('cuda')
                    logits,states = model(enc_text)
                    last_states = get_indexes(enc_text, states)
                    #last_states = states[:,-1,:]
                    classification_out = regressionhead(last_states)
                    #print(classification_out)
                    #probs = F.softmax(classification_out,dim=-1)
                    #print(probs)
                    if NUM_CLASSES>1:
                        preds = torch.argmax(classification_out,dim=-1).squeeze()
                    else:
                        preds = (classification_out).squeeze()*5.0
                    #print(preds.shape,enc_labels.shape)
                    assert preds.shape == enc_labels.shape 
                    diff.append(torch.mean(torch.abs(preds-enc_labels)))
                    predictions.extend(preds.tolist())
                    tgts.extend(enc_labels.tolist())
                    labels.extend(preds.tolist())
            print(f'Labels = {sum(labels)/len(labels)}')
            print(f'Diff = {sum(diff)/len(diff)}')
            print(f'Pearson = {pearson(torch.tensor(predictions)/5.0,torch.tensor(tgts)/5.0)}')
    print(f'Epoch {epoch} completed')
print('Finished training')
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
torch.save(regressionhead,checkpoint_dir+'head-last.ckpt')
torch.save(model,checkpoint_dir+'model-last.ckpt')
print('\n'*5)
print(TASK_NAME,NUM_CLASSES,MODEL_NAME, BATCH_SIZE,EPOCHS, 'COMPLETED')
print('\n'*5)



