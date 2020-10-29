import nltk
import pandas as pd

nltk.download('punkt')

from nltk.tokenize import word_tokenize

df = pd.read_csv('data/ted_talks.csv')
ted_talks = df[['description', 'title']]

print(word_tokenize("salam khoobi"))

for index, row in ted_talks.iterrows():
    row['description'] = word_tokenize(row['description'].lower())
    row['title'] = word_tokenize(row['title'].lower())

print(ted_talks)
