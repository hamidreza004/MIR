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
stop_word_ratio = 0.0026
stop_words = []


def clean_raw(raw):
    tokens = word_tokenize(raw.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [snowball.stem(word) for word in tokens]
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
    return tokens


def remove_stop_words(tokens):
    return [word for word in tokens if not word in stop_words]


def read_ted_file(path):
    df = pd.read_csv(path)
    ted_df = df[['description', 'title']]
    return ted_df


def find_stop_words(all_tokens_dic):
    stop_point = len(all_tokens_dic.items()) * stop_word_ratio
    i = 0
    print("Stopwords are ...:")
    for token, count in sorted(all_tokens_dic.items(), key=lambda item: item[1], reverse=True):
        i += 1
        stop_words.append(token)
        print(token, count)
        if i > stop_point:
            break


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

    find_stop_words(all_tokens_dic)

    for index, row in df.iterrows():
        row['description'] = remove_stop_words(row['description'])
        row['title'] = remove_stop_words(row['title'])

    return df


ted_talks = prepare_text(read_ted_file('data/ted_talks.csv'))
print(ted_talks)
