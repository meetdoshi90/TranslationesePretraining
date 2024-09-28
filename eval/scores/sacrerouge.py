import pandas as pd
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rouge3','rouge4', 'rougeL'], use_stemmer=True, lang="hindi")

FILENAME = '/raid/nlp/pranavg/meet/IndicLLM/IndicGPT/eval/scores/MiniLM-Hi-small-IndicHeadlineGeneration.csv'

data = pd.read_csv(FILENAME)

target = data['target']
predictions = data['generated']
target = [[i] for i in target]

scores = scorer.score('वाराणसी में दवा कारोबारियों पर सख्ती',['वाराणसी में दवा कारोबारियों पर सख्ती'])

print(scores)
