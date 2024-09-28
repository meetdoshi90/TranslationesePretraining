import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER
bleu = BLEU()
chrf = CHRF()
ter = TER()

FILENAME = 'MiniLM-BI-en-hi-small-en-hi-IN22-Conv.csv'

data = pd.read_csv(FILENAME)

target = data['target']
target = [[i] for i in target]
predictions = data['generated']

bleu_results = bleu.corpus_score(predictions=predictions,references=target)
print(bleu_results)

chrf_results = chrf.corpus_score(predictions,target)
print(chrf_results)

ter_results = ter.corpus_score(predictions,target)
print(ter_results)