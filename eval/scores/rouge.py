import evaluate
import pandas as pd
rouge_score = evaluate.load("rouge")

FILENAME = 'MiniLM-Hi-small-IndicHeadlineGeneration.csv'

data = pd.read_csv(FILENAME)

target = data['target']
predictions = data['generated']

results = rouge_score.compute(predictions=predictions,references=target)
print(results)