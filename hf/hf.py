import torch
import sentencepiece as spm
torch.backends.cuda.enable_flash_sdp(True) #Enable flash scaled dot product attention
torch.backends.cuda.enable_mem_efficient_sdp(True) #Enable mem efficient SDP
torch.backends.cuda.enable_math_sdp(True) #Math sdp
#Print status
print(torch.backends.cuda.flash_sdp_enabled())
print(torch.backends.cuda.mem_efficient_sdp_enabled())
print(torch.backends.cuda.math_sdp_enabled())
from customgpt2 import MiniLMHeadModel
from transformers import AutoTokenizer
from originalgpt2 import GPTModel
from configs import Config

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


OGMODEL_NAME = 'ft-hi-small-IndicHeadlineGeneration'
HFMODEL_NAME = 'MiniLM-Hi-small-IndicHeadlineGeneration'

model_path = f'/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints-{OGMODEL_NAME}/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'


config = Config()
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

model = MiniLMHeadModel(config=config)

ogmodel = GPTModel.load_from_checkpoint(model_path)

print(type(model))
#print(dir(model))
total = 0
for n,p in model.named_parameters():
    print(n,p.shape)
    total += p.numel()
print('Total params in new model', total)
total = 0
for n,p in ogmodel.named_parameters():
    print(n,p.shape)
    total += p.numel()
print('Total params in old model', total)


old_state_dict = ogmodel.state_dict()

old_key = 'model.lm_head.weight'
new_key = 'lm_head.weight'
old_state_dict[new_key] = old_state_dict.pop(old_key)

old_key = 'model.lm_head.bias'
new_key = 'lm_head.bias'
old_state_dict[new_key] = old_state_dict.pop(old_key)

model.load_state_dict(old_state_dict)

inps = tokenizer.encode(
    '''उदय चौधरी/जयपुर।बेरोजगार युवाओं से नौकरी के नाम पर लाखों रूपये ठगने वाले गिरोह के एक और सदस्य को एसओजी ने गिरफ्तार किया है।अतिरिक्त महानिदेशक पुलिस, एटीएस एवं एसओजी अनिल पालीवाल ने बताया कि गिरफ्तार आरोपी हुकमचन्द मीना पुत्र मोहन लाल (37) निवासी अम्बेडकर नगर, थाना सदर जिला अलवर है।बेरोजगारों युवाओं को नौकरी दिलवाने के नाम पर लाखों रूपये की ठगी करने वाले गिरोह का एसओजी ने शनिवार को पर्दाफाश कर रेनवाल नगरपालिका के पूर्व अध्यक्ष व गिरोह के मुख्य सरगना हरिप्रकाश तोतला सहित तीन सदस्यों को गिरफ्तार किया गया था।इसी गिरोह का एक अन्य सदस्य हुकमचन्द मीना प्रकरण दर्ज होने की सूचना मिलते ही फरार हो गया था।एडीजी पालीवाल ने बताया कि रविवार रात को मुखबिर से आरोपी हुकुम चंद के चौमूं आने की सूचना मिली थी।एसओजी की टीम ने गिरफ्तार कर लिया।उन्होंने बताया कि वर्तमान में आरोपी हुकुमचंद वर्तमान में एफसीआई नई दिल्ली में असिस्टेंट ग्रेड-2 के पद पर पदस्थापित है।आरोपी द्वारा कितने युवाओं को इस प्रकार नौकरी का झांसा देकर अपना शिकार बनाया है, इस संबंध में गहन अनुसंधान जारी है।'''
)

inps = [1] + inps + [2]

print(inps)

out = model.generate(
    torch.tensor(inps).view(1,-1),
    num_beams=5,
    max_new_tokens = 20,
    no_repeat_ngram_size=5,
    repetition_penalty=1.0
)

out = out[0].tolist()

print(out)

print(tokenizer.decode(out))

Config.model_type = HFMODEL_NAME
MiniLMHeadModel.config_class = Config
AutoConfig.register(HFMODEL_NAME,Config)
#AutoModel.register(config, model)
AutoModelForCausalLM.register(Config,MiniLMHeadModel)

#config.register_for_auto_class()
#model.register_for_auto_class('AutoModelForCausalLM')

from huggingface_hub import notebook_login,login

login(token='INSERT_TOKEN_HERE')
#notebook_login()

model.push_to_hub('MiniLM-Hi-small-IndicHeadlineGeneration')
