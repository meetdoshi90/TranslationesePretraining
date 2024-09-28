import torch
import sentencepiece as spm
import gradio as gr
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

VOCAB_SIZE = 56000

choices = {
    'as':'asm_Beng',
    'bd':'brx_Deva',
    'bn':'ben_Beng',
    'dg':'doi_Deva',
    'en':'eng_Latn',
    'gom':'gom_Deva',
    'gu':'guj_Gujr',
    'hi':'hin_Deva',
    'kn':'kan_Knda',
    'ks':'kas_Arab',
    'mai':'mai_Deva',
    'ml':'mal_Mlym',
    'mni':'mni_Mtei',
    'mr':'mar_Deva',
    'ne':'npi_Deva',
    'or':'ory_Orya',
    'pa':'pan_Guru',
    'sa':'san_Deva',
    'sat':'sat_Olck',
    'sd':'snd_Arab',
    'ta':'tam_Taml',
    'te':'tel_Telu',
    'ur':'urd_Arab',
}




model_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/checkpoints-en-tiny_4096/last.ckpt'
tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'


model = GPTModel.load_from_checkpoint(model_path)
model.eval()
#model.half()
#print(dir(model))

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
print('Loaded models...')

class Config:
    context_len = 1024
config = Config()

def generate(model, prompt, max_tokens=30, temperature=1.0, config=None,num_beams=5, len_penalty=1.0, penalise_n_grams=4):
    """
    Generates text based on the provided prompt.
    Model determinism can be changed with temperature (range: [0, 1], higher means more unstable but creative predictions)
    """
    model.eval()
    '''
    for _ in range(max_tokens):
        prompt = prompt
        #print(prompt)
        logits = model(prompt)
        #print(len(logits),logits[0].shape,logits[1].shape)
        logits = logits[:, -1, :] / temperature
        logit_probs = torch.nn.functional.softmax(logits, dim=-1)
        #next_prompt = torch.multinomial(logit_probs, num_samples=1)
        next_prompt = torch.topk(logit_probs, k=1, dim=-1).indices.view(-1,1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt
    '''
    inp_len = prompt.shape[1]
    beams = []
    probs = []
    num_beams = int(num_beams)
    penalise_n_grams = int(penalise_n_grams)
    mask = torch.ones((num_beams,1)).to('cuda')
    for i in range(max_tokens):
        if i==0:
            logits,x = model(prompt)
            logits = logits[:, -1, :] / temperature
            logit_probs = torch.nn.functional.softmax(logits, dim=-1).log()
            top_indices = logit_probs.topk(k=num_beams)
            probs = top_indices.values.view(-1,1)
            beams = top_indices.indices.view(-1,1)
            prompt = prompt.repeat(num_beams,1)
            prompt = torch.cat([prompt,beams],dim=1)
        else:
            #lp_y = (5.0 + i + 1)**len_penalty / (5.0 + 1.0)**len_penalty
            sequence = torch.arange(i).repeat(num_beams,1).to('cuda')
            seq_mask = prompt[:,inp_len:]
            seq_mask = torch.where(seq_mask==2,1.0,0.0)
            sequence = sequence.where(seq_mask.bool(),0.0).sum(dim=1)
            sequence = torch.where(sequence!=0.0,sequence,i)
            lp_y = ((5.0 + sequence + 1)**len_penalty / (5.0 + 1.0)**len_penalty).view(-1,1) #len_penalty == alpha
            
            logits,x = model(prompt)
            logits = logits[:, -1, :] / temperature
            logit_probs = torch.nn.functional.softmax(logits, dim=-1)
            logit_probs = logit_probs.where(mask.bool(),1.0).log()
            logit_probs = logit_probs/lp_y #Length normalisation
            logit_probs = probs + logit_probs # P(prev,V | inp) = P(prev | inp) * P(V | inp, prev)
            B, V = logit_probs.shape
            top_indices = logit_probs.view(B*V).topk(k=num_beams)
            indices = top_indices.indices 
            probs = top_indices.values.view(-1,1)

            eos = ((indices%VOCAB_SIZE)!=2).view(-1,1)
            mask = torch.logical_and(mask, eos)

            select_beams = indices // VOCAB_SIZE
            new_beams = prompt.index_select(0,select_beams)
            new_beams = torch.cat([new_beams,(indices%VOCAB_SIZE).view(-1,1)],dim=1)
            prompt = new_beams
    return prompt
    #''' 
            






def preprocess(text,choice):
    en_tok = MosesTokenizer(lang="en")
    en_normalizer = MosesPunctNormalizer()
    en_detok = MosesDetokenizer(lang="en")
    xliterator = unicode_transliterate.UnicodeIndicTransliterator()

    for i in choices:
        if choice==i:
            lang = choices[i]

    iso_lang = flores_codes[lang]
    text = punc_norm(text, iso_lang)
    if lang == "eng_Latn":
        normalizer = None
    else:
        normfactory = IndicNormalizerFactory()
        normalizer = normfactory.get_normalizer(flores_codes[lang])
    
    transliterate = True
    if lang.split("_")[1] in ["Arab", "Aran", "Olck", "Mtei", "Latn"]:
        transliterate = False
    
    if iso_lang == "en":
        text = " ".join(
            en_tok.tokenize(
                en_normalizer.normalize(text.strip()), escape=False
            )
        )
    elif transliterate:
        # transliterates from the any specific language to devanagari
        # which is why we specify lang2_code as "hi".
        text = xliterator.transliterate(
            " ".join(indic_tokenize.trivial_tokenize(normalizer.normalize(text.strip()), iso_lang)),
            iso_lang,
            "hi",
        ).replace(" ् ", "्")
    else:
        # we only need to transliterate for joint training
        text = " ".join(
            indic_tokenize.trivial_tokenize(normalizer.normalize(text.strip()), iso_lang)
        )
    return text

def postprocess(text,choice):
    en_tok = MosesTokenizer(lang="en")
    en_normalizer = MosesPunctNormalizer()
    en_detok = MosesDetokenizer(lang="en")
    xliterator = unicode_transliterate.UnicodeIndicTransliterator()
    for i in choices:
        if choice==i:
            lang = choices[i]

    iso_lang = flores_codes[lang]
    text = punc_norm(text, iso_lang)
    lang_code, script_code = lang.split('_')
    print(lang)
    if lang == "eng_Latn":
        text = en_detok.detokenize(text.split(" "))
    else:
        text = indic_detokenize.trivial_detokenize(
            xliterator.transliterate(text, flores_codes['hin_Deva'], flores_codes[lang]), flores_codes[lang]
        )
    return text

def gradio_inp(text=None,choice=None,max_tokens=None,temperature=None, num_beams=None, len_penalty=None, penalise_n_grams=None):
    global model
    global config
    max_tokens = int(max_tokens)
    text = preprocess(text,choice)
    #temp = 's'
    #text = '<'+temp+'> ' + text
    print(text)
    enc_inp = [1] + tokenizer.encode(text) # + [2]
    #enc_inp = [2] * 10
    inp_len = len(enc_inp)
    enc_inp = torch.tensor(enc_inp).view(1,-1).to('cuda')
    print(tokenizer.decode(enc_inp.view(-1,1).tolist()))
    out = generate(model,enc_inp,max_tokens,temperature,config, num_beams, len_penalty,penalise_n_grams)
    print(out)
    out = out[:,inp_len:].tolist()[0]
    temp = []
    for o in out:
        if o==2:
            break
        else:
            temp.append(o)
    out = temp
    #print(out)
    out = tokenizer.decode(out)
    print(out)
    out = postprocess(out,choice)
    return out

examples = [
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"],
]


demo = gr.Interface(
    fn=gradio_inp,
    inputs=[gr.inputs.Textbox(lines=5, label="Input Text"), 
            gr.Dropdown(choices=choices.keys(),max_choices=1), 
            gr.Number(value=10,label='Max tokens'), 
            gr.Number(value=0.7, precision=5,label='Temperature'),
            gr.Number(value=5,label='Num beams'), 
            gr.Number(value=1.0, precision=5,label='Length Penalty'),
            gr.Number(value=5,label='N gram penalty'), 
            ],
    outputs="text",
    examples=examples
)

if __name__ == "__main__":
    demo.launch()
    # out = gradio_inp(
    #     'टाटा कारों की रेंज में सबसे ज्यादा डिस्काउंट सफारी स्टॉर्म (एक लाख रुपए) पर मिल रहा है, वहीं सबसे कम डिस्काउंट टाटा टियागो (27 हजार रुपए) पर है।',
    #     'hi',
    #     64,
    #     0.8,
    #     5,
    #     1.0,
    #     4
    # )
    # print(out)