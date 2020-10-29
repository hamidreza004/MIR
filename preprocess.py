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


def read_ted_file(path):
    df = pd.read_csv(path)
    ted_df = df[['description', 'title']]
    return ted_df


def clean_file(df):
    for index, row in df.iterrows():
        row['description'] = clean_raw(row['description'])
        row['title'] = clean_raw(row['title'])
    return df


ted_talks = clean_file(read_ted_file('data/ted_talks.csv'))
print(ted_talks)
