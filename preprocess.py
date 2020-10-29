import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

snowball = SnowballStemmer("english")
lemma = WordNetLemmatizer()


def clean_raw(raw):
    tokens = word_tokenize(raw.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [snowball.stem(word) for word in tokens]
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]

    return tokens


def read_ted_file(path):
    df = pd.read_csv(path)
    ted_df = df[['description', 'title']]
    return ted_df


def prepare_text(df):
    all_tokens_dic = {}
    for index, row in df.iterrows():
        row['description'] = clean_raw(row['description'])
        row['title'] = clean_raw(row['title'])
        for col_name in ['description', 'title']:
            for token in row[col_name]:
                if not token in all_tokens_dic:
                    all_tokens_dic[token] = 1
                else:
                    all_tokens_dic[token] += 1
    i = 0
    for token, count in sorted(all_tokens_dic.items(), key=lambda item: item[1], reverse=True):
        i += 1
        print(token, count)
        if i == 100:
            break
    return df


ted_talks = prepare_text(read_ted_file('data/ted_talks.csv'))
print(ted_talks)
