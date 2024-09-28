import evaluate
import pandas as pd
bleu_score = evaluate.load("bleu")

FILENAMES = [   
'MiniLM-BI-en-hi-small-en-hi-IN22-Conv.csv',
'MiniLM-BI-en-hi-small-en-hi-IN22-Gen.csv',
'MiniLM-BI-en-hi-small-hi-en-IN22-Conv.csv',
'MiniLM-BI-en-hi-small-hi-en-IN22-Gen.csv',
'MiniLM-BI-en-hi_syn-diffsrc-small-en-hi-IN22-Conv.csv',
'MiniLM-BI-en-hi_syn-diffsrc-small-en-hi-IN22-Gen.csv',
'MiniLM-BI-en-hi_syn-diffsrc-small-hi-en-IN22-Conv.csv',
'MiniLM-BI-en-hi_syn-parallel-small-en-hi-IN22-Conv.csv',
'MiniLM-BI-en-hi_syn-parallel-small-en-hi-IN22-Gen.csv',
'MiniLM-BI-en-hi_syn-parallel-small-hi-en-IN22-Conv.csv',
'MiniLM-BI-en-hi_syn-parallel-small-hi-en-IN22-Gen.csv'
]
for FILENAME in FILENAMES:
    data = pd.read_csv(FILENAME)

    target = data['target']
    predictions = data['generated']

    results = bleu_score.compute(predictions=predictions,references=target)
    print(FILENAME)
    print(results)