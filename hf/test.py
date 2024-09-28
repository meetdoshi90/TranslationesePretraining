from transformers import AutoModelForCausalLM, AutoTokenizer
import sentencepiece as spm
import torch

tokenizer_path = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/models/ACL24_Multi_8.model'
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

model = AutoModelForCausalLM.from_pretrained('meetdoshi90/MiniLM-Hi-small-IndicHeadlineGeneration',trust_remote_code=True)

inps = tokenizer.encode(
    '''उदय चौधरी/जयपुर।बेरोजगार युवाओं से नौकरी के नाम पर लाखों रूपये ठगने वाले गिरोह के एक और सदस्य को एसओजी ने गिरफ्तार किया है।अतिरिक्त महानिदेशक पुलिस, एटीएस एवं एसओजी अनिल पालीवाल ने बताया कि गिरफ्तार आरोपी हुकमचन्द मीना पुत्र मोहन लाल (37) निवासी अम्बेडकर नगर, थाना सदर जिला अलवर है।बेरोजगारों युवाओं को नौकरी दिलवाने के नाम पर लाखों रूपये की ठगी करने वाले गिरोह का एसओजी ने शनिवार को पर्दाफाश कर रेनवाल नगरपालिका के पूर्व अध्यक्ष व गिरोह के मुख्य सरगना हरिप्रकाश तोतला सहित तीन सदस्यों को गिरफ्तार किया गया था।इसी गिरोह का एक अन्य सदस्य हुकमचन्द मीना प्रकरण दर्ज होने की सूचना मिलते ही फरार हो गया था।एडीजी पालीवाल ने बताया कि रविवार रात को मुखबिर से आरोपी हुकुम चंद के चौमूं आने की सूचना मिली थी।एसओजी की टीम ने गिरफ्तार कर लिया।उन्होंने बताया कि वर्तमान में आरोपी हुकुमचंद वर्तमान में एफसीआई नई दिल्ली में असिस्टेंट ग्रेड-2 के पद पर पदस्थापित है।आरोपी द्वारा कितने युवाओं को इस प्रकार नौकरी का झांसा देकर अपना शिकार बनाया है, इस संबंध में गहन अनुसंधान जारी है।'''
)

inps = [1] + inps + [2]

print(inps)

out = model.generate(
    torch.tensor(inps).view(1,-1),
    num_beams=5,
    max_new_tokens = 70,
    no_repeat_ngram_size=5,
    length_penalty=1.0
)
print(out)
print(tokenizer.decode(out[0].tolist()))