import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import lightning.pytorch as pl
from models.rope import RotaryEmbedding, rotate_half, apply_rotary_pos_emb
#from xformers.components.positional_embedding import RotaryEmbedding
# import wandb
# from wandb_setup import wandb_log
from time import time

class CausalSelfAttentionHead(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttentionHead, self).__init__()
        self.config = config

        self.query = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.attn_drop = nn.Dropout(config.attn_drop_value)
        self.rope = RotaryEmbedding(config.head_size)
        # if self.config.positional_embedding=='rope':
        #     self.rotary_emb = RotaryEmbedding(dim_model = self.config.head_size)
        # else:
        #     self.rotary_emb = None
        self.register_buffer('tril', torch.tril(torch.ones(config.context_len, config.context_len)))

    def forward(self, x):
        # print('Causal self attention called')
        # x.shape: (Batch, Context Length, Embedding Dimension)
        B, C, N = x.shape
        q = self.query(x) # (B, C, head_size)
        k = self.key(x) # (B, C, head_size)
        v = self.value(x) # (B, C, head_size)
        # if self.config.positional_embedding=='rope':
        #     print(q.shape, k.shape)
        #     q,k = self.rotary_emb(q,k)
        #     print(q.shape, k.shape)

        # if self.config.use_flashattn:
            #start = time()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            q = q.view(q.shape[0],1,q.shape[1],q.shape[2])
            k = k.view(k.shape[0],1,k.shape[1],k.shape[2])
            v = v.view(v.shape[0],1,v.shape[1],v.shape[2])
            q,k = self.rope(q,k)
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.config.attn_drop_value, is_causal=True).squeeze()
        #print('Flash attention took {} seconds'.format(time() - start))
        # else:
        #     # Compute Attention scores
        #     # (B, C, head_size) bmm (B, head_size, C) -> (B, C, C)
        #     #start = time()
        #     attn_weight = torch.div(torch.bmm(q, k.permute(0, 2, 1)), self.config.head_size)
        #     attn_weight = attn_weight.masked_fill(self.tril[:C, :C] == 0, float('-inf'))
        #     attn_weight = F.softmax(attn_weight, dim=-1)
        #     attn_weight = self.attn_drop(attn_weight)

        #     # Do weighted aggregation of values
        #     output = torch.bmm(attn_weight, v)
            #print('Standard attention took {} seconds'.format(time() - start))
        return output

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.embed_dim = config.n_embed
    
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(config) for _ in range(self.num_heads)]
        )
        self.proj = nn.Linear(config.num_heads * config.head_size, config.n_embed)
        self.drop = nn.Dropout(config.multihead_drop_value)
        self.config = config

    def forward(self, x):
        multihead_output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.drop(self.proj(multihead_output))
    
class CustomMultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(CustomMultiHeadedAttention, self).__init__()
        assert config.n_embed % config.num_heads == 0
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.embed_dim = config.n_embed

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # QKV
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        print('Using flash attention yes or no?',self.flash)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.rope = RotaryEmbedding(config.head_size)
        self.config = config

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q,k = self.rope(q,k)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    
class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * config.ffn_scaling),
            nn.GELU(),
            nn.Linear(config.n_embed * config.ffn_scaling, config.n_embed),
            nn.Dropout(config.ffn_drop_value)
        )
    def forward(self, x):
        return self.ffn(x)
    
class GPTBlock(nn.Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        self.multiheaded_attn = CustomMultiHeadedAttention(config)
        self.ffn = FFN(config)
        self.layernorm1 = nn.LayerNorm(config.n_embed)
        self.layernorm2 = nn.LayerNorm(config.n_embed)
    
    def forward(self, x):
        x = x + self.layernorm1(self.multiheaded_attn(x))
        x = x + self.layernorm2(self.ffn(x))
        return x
    
class GPT(pl.LightningModule):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        # Init layers and stuff
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        # if self.config.positional_embedding=='rope':
        #     self.pos_embedding = None
        # else:
        #self.pos_embedding = nn.Embedding(config.context_len, config.n_embed)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.num_blocks)])
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        #self.pos_emb = self.pos_embedding(torch.arange(self.config.context_len, device=self.device))
    
    def forward(self, x):
        # Input is just tokenized text of 'B' batches, each 'C' context length long
        B, C = x.shape
        # First we apply the token embedding -> tok_emb (B, C, V)
        tok_emb = self.tok_embedding(x)
        # if self.config.positional_embedding=='rope':
        #     x = tok_emb
        # else:
        #pos_emb = self.pos_embedding(torch.arange(C, device=self.device))
        x = tok_emb 
        x = self.blocks(x)
        # And finally pass it through the final layer to get the logits
        logits = self.lm_head(x)
        return logits
    
def generate(model, prompt, max_tokens=10, temperature=0.7,config=None):
    model.eval()
    for _ in range(max_tokens):
        prompt = prompt[:, :config.context_len]
        logits = model(prompt)
        logits = logits[:, -1, :] / temperature
        logit_probs = nn.functional.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt


class GPTModel(pl.LightningModule):
    def __init__(self, config,tokenizer):
        super(GPTModel, self).__init__()
        self.save_hyperparameters()
        # Model Architecture
        self.config = config
        self.model = GPT(self.config)
        self.tokenizer = tokenizer
    
    def get_loss(self, logits, targets):
        #print(logits.shape)
        B, C, V = logits.shape
        logits = logits.view(B*C, V)
        targets = targets.view(B*C)
        PAD_ID = self.config.PAD_TOKEN_ID
        #UNK_ID = self.config.UNK_TOKEN_ID
        loss = nn.functional.cross_entropy(logits, targets,reduction='none')
        assert loss.shape==targets.shape
        loss_mask_1 = targets!=PAD_ID
        #loss_mask_2 = targets!=UNK_ID
        loss_masked = loss.where(loss_mask_1, torch.tensor(0.0))
        #loss_masked = loss_masked.where(loss_mask_2, torch.tensor(0.0))
        return loss_masked.sum()/torch.count_nonzero(loss_masked)
    
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        text, target = batch
        text = text.long()
        target = target.long()
        logits = self(text)
        loss = self.get_loss(logits, target)
        self.log(f'train_loss',loss)
        #logs = {'loss': loss}
        # if self.config.wandb:
        #     wandb_log(loss=loss.item())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        #print('\nValidation',batch_idx, dataloader_idx,'\n')
        #print(torch.cuda.mem_get_info())
        text, target = batch
        text = text.long()
        target = target.long()
        with torch.no_grad():
            logits = self(text)
        loss = self.get_loss(logits, target)
        self.log(f'{self.config.val_id_to_name[dataloader_idx]}_val_loss',loss)
        #logs = {'loss': loss}
        # if self.config.wandb:
        #     wandb_log(loss=loss.item())
        #generated_text = generate(self.model,text[:2,:],5,0.95,self.config)
        #print('\n',f'Generated Text {self.config.val_id_to_name[dataloader_idx]}','\n')
        #print(generated_text.tolist())
        #print(generated_text)
        #print(self.tokenizer.decode(generated_text.tolist()))
        #print(target[:2,:])
        #print('\n',f'Actual Text {self.config.val_id_to_name[dataloader_idx]}','\n')
        #print(target[:2,:].tolist())
        #print(self.tokenizer.decode(target[:2,:].tolist()))
        return loss
    
    
    # def training_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'].mean() for x in outputs]).mean()
        
    #     #logs = {'loss': avg_loss}
    #     # if self.config.wandb:
    #     #     wandb_log(avg_loss=avg_loss)
        
    #     #print(f"val_loss: {avg_loss}")
    #     return avg_loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config.lr, betas= (self.config.beta_1,self.config.beta_2), eps=self.config.eps, weight_decay=self.config.wd)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self.lr_warmup_steps)
        return [opt], []