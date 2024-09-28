from minilm.configuration_minilm import Config
from minilm.modeling_minilm import MiniLMHeadModel

import sentencepiece as spm
import torch

from originalgpt2 import GPTModel
#TASK = 'IndicSentenceSummarization'
#TASK = 'IndicQuestionGeneration'
#TASK = 'IndicHeadlineGeneration'
#TASK = 'IndicParaphrase'
#TASK = 'IndicWikiBio'
#TASK = 'xlsum-headline'
#TASK = 'xlsum-summarization'
#TASK = 'dialogsum'
#TASK = 'cnn_dailymail'
TASK = 'extended-syn-hi-IndicWikiBio'
TRAINTYPE = 'ft-syn-hi'
SIZE = 'small'
OGMODEL_NAME = f'ft-{TRAINTYPE}-{SIZE}-{TASK}'
#OGMODEL_NAME = f'{TRAINTYPE}-{SIZE}'
HFMODEL_NAME = f'MiniLM-{TRAINTYPE}-{SIZE}-{TASK}'
#HFMODEL_NAME = f'MiniLM-base-{TRAINTYPE}-{SIZE}'

print(OGMODEL_NAME,'pushing to ...', HFMODEL_NAME)

model_path = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints-{OGMODEL_NAME}/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'


tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
ogmodel = GPTModel.load_from_checkpoint(model_path)
old_state_dict = ogmodel.state_dict()

old_key = 'model.lm_head.weight'
new_key = 'lm_head.weight'
old_state_dict[new_key] = old_state_dict.pop(old_key)

old_key = 'model.lm_head.bias'
new_key = 'lm_head.bias'
old_state_dict[new_key] = old_state_dict.pop(old_key)



Config.model_type = HFMODEL_NAME
MiniLMHeadModel.config_class = Config
Config.register_for_auto_class()
MiniLMHeadModel.register_for_auto_class("AutoModelForCausalLM")

new_model_config = Config()
new_model = MiniLMHeadModel(new_model_config)
new_model.config_class = Config

new_model.load_state_dict(old_state_dict)
print(new_model.config_class)

from huggingface_hub import notebook_login,login

login(token='INSERT_TOKEN_HERE')
#notebook_login()

new_model.push_to_hub(HFMODEL_NAME)

