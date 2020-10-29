import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

snowball = SnowballStemmer("english")


def clean_raw(raw):
    tokens = word_tokenize(raw.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    tokens = [snowball.stem(word) for word in tokens]
    return tokens


df = pd.read_csv('data/ted_talks.csv')
ted_talks = df[['description', 'title']]

for index, row in ted_talks.iterrows():
    row['description'] = clean_raw(row['description'])
    row['title'] = clean_raw(row['title'])

print(ted_talks)
