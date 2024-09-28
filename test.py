import torch
import sentencepiece as spm
torch.backends.cuda.enable_flash_sdp(True) #Enable flash scaled dot product attention
torch.backends.cuda.enable_mem_efficient_sdp(True) #Enable mem efficient SDP
torch.backends.cuda.enable_math_sdp(True) #Math sdp
#Print status
print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
from models.gpt2_rope_inf import GPTModel
import lightning.pytorch as pl

model_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/Extra_Merged_SS.model'

model = GPTModel.load_from_checkpoint(model_path)
model.eval()

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
print('Loaded models...')

sentence = 'Attention to details \n \n is uncompromisable'
print('input sentence', sentence)

enc_inp = [1] + tokenizer.encode(sentence) 

enc_inp = torch.tensor(enc_inp).view(1,-1).to('cuda')

print(tokenizer.decode(enc_inp.view(-1,1).tolist()))
print(enc_inp)

class Config:
    context_len = 96
config = Config()

def generate(model, prompt, max_tokens=10, temperature=0.7,config=None):
    """
    Generates text based on the provided prompt.
    Model determinism can be changed with temperature (range: [0, 1], higher means more unstable but creative predictions)
    """
    model.eval()
    for _ in range(max_tokens):
        prompt = prompt[:, :config.context_len]
        logits = model(prompt)
        logits = logits[:, -1, :] / temperature
        logit_probs = torch.nn.functional.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt

def generate_top_k(model, prompt, top_k = 10, temperature=0.99,config=None):
    """
    Generates text based on the provided prompt.
    Model determinism can be changed with temperature (range: [0, 1], higher means more unstable but creative predictions)
    """
    model.eval()
    prompt = prompt[:, :config.context_len]
    logits = model(prompt)
    logits = logits[:, -1, :] / temperature
    logit_probs = torch.nn.functional.softmax(logits, dim=-1)
    next_prompt = torch.multinomial(logit_probs, num_samples=10)
    #prompt = torch.cat((prompt, next_prompt), dim=1)
    return next_prompt.view(-1,1)

generated_out = generate(model,enc_inp,max_tokens=20,temperature=1.0,config=config)
#generated_out = generate_top_k(model,enc_inp,top_k=5,temperature=0.99,config=config)

print(generated_out)
print(tokenizer.decode(generated_out.tolist()))