import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm

FOLDER = 'EN-scores'
files = os.listdir(FOLDER)
if not os.path.exists('./plots/'):
    os.mkdir('./plots/')

sequence = []
position = []
score = []

for file in tqdm(files):
    if 'json' in file:
        with open(f'./{FOLDER}/{file}') as f:
            data = json.load(f)
            random.shuffle(data)
            data = data[:1000]
            for i in range(len(data)-1):
                dic = data[i]
                if len(dic['scores'])>10000:
                    continue
                j = 0
                for sc in dic['scores']:
                    if j > len(dic['tokens']):
                        continue
                    sequence.append(file.split('.')[0])
                    position.append(j)
                    score.append(sc)
                    j+=1
        #break

df = pd.DataFrame(
    {
        'Position': position,
        'Score': score,
        'Sequence': sequence
    }
)

print(df.head())
df.to_csv('plot.csv',index=False)

plt.figure(figsize=(48, 24))
sns.violinplot(x='Position', y='Sequence', data=df)
plt.title('Violin Plot of Score Variance Across Positions')
plt.savefig(f'./plots/{FOLDER}-violin.png')
plt.clf()
SEQ = 'OSCAR_2301_en_meta_part_1'
df1 = df.loc[df['Sequence']==SEQ]
df1['Position'] = df1['Position'].astype(int)
plt.figure(figsize=(256, 24))
sns.boxplot(data=df1, x='Position', y='Score', orient='v')
plt.title('Box Plot of Score Variance Across Positions')
plt.xticks(rotation=90)
plt.savefig(f'./plots/{FOLDER}-box-{SEQ}.png')
plt.clf()

