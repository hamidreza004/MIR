import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer

df = pd.read_csv('data/ted_talks.csv')
ted_talks = df[['description', 'title']]

for index, row in ted_talks.iterrows():
    row['description'] = row['description'].lower()
    row['title'] = row['title'].lower()

porter = PorterStemmer()
for stem in ["automatic", "automatically", "swears",
             "swore", "presidential", "president"]:
    print(porter.stem(stem))
