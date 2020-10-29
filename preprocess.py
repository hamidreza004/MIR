import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

snowball = SnowballStemmer("english")
lemma = WordNetLemmatizer()


def clean_raw(raw):
    tokens = word_tokenize(raw.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stopwords.words("english")]
    tokens = [snowball.stem(word) for word in tokens]
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]

    return tokens


df = pd.read_csv('data/ted_talks.csv')
ted_talks = df[['description', 'title']]

for index, row in ted_talks.iterrows():
    row['description'] = clean_raw(row['description'])
    row['title'] = clean_raw(row['title'])

print(ted_talks)
