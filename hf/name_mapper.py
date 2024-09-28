import os
files = os.listdir('/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/')

for file in files:
    if 'checkpoints-' in file:
        print(file)