import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
import nltk

nltk.download('punkt')

df = pd.read_csv('data/ted_talks.csv')
ted_talks = df[['description', 'title']]

print(sent_tokenize("salam khoobi"))

for index, row in ted_talks.iterrows():
    row['description'] = sent_tokenize(row['description'].lower())
    row['title'] = sent_tokenize(row['title'].lower())

print(ted_talks)
