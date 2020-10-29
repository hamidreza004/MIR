import pandas as pd

df = pd.read_csv('data/ted_talks.csv')
ted_talks = df[['description', 'title']]

for index, row in ted_talks.iterrows():
    row['description'] = row['description'].lower()
    row['title'] = row['title'].lower()

print(ted_talks)
