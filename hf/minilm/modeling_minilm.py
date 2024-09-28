import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
from typing import List, Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .configuration_minilm import Config

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    # NOTE: This could probably be moved to Triton

    # Handle a possible sequence length mismatch in between q and k
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[seq_dimension], device=x.device, dtype=torch.float32
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :].to(x.dtype)
            self._sin_cached = emb.sin()[None, None, :, :].to(x.dtype)

        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
    


class MiniLMPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range if self.config.initializer_range else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
   
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

        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q,k = self.rope(q,k)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
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
    
class MiniLM(MiniLMPreTrainedModel):
    #config_class = None
    def __init__(self, config):
        super(MiniLM,self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embed, self.padding_idx)
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.num_blocks)])
        
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids")

        inputs_embeds = self.tok_embedding(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        hidden_states = self.blocks(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states,] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


class MiniLMHeadModel(MiniLMPreTrainedModel):
    config_class = Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(MiniLMHeadModel,self).__init__(config)
        # Model Architecture
        self.config = config
        self.model = MiniLM(self.config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            labels_shape = shift_labels.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # Return reshaped loss
            loss = loss.view(labels_shape)

        if not return_dict:
            output = (logits,) 
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            raise KeyError('past key values not implemented')
        

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        return model_inputs
