import nltk
from preprocess.stopwords import remove_stop_words, find_stop_words

nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

snowball = SnowballStemmer("english")
lemma = WordNetLemmatizer()
stop_word_ratio = 0.003
stop_words = []


def tokenize_raw(raw):
    return word_tokenize(raw.lower())


def clean_raw(raw):
    tokens = tokenize_raw(raw)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [snowball.stem(word) for word in tokens]
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
    return tokens


def prepare_text(df):
    global stop_words
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

    stop_words = find_stop_words(all_tokens_dic, stop_word_ratio)

    for index, row in df.iterrows():
        row['description'] = remove_stop_words(row['description'], stop_words)
        row['title'] = remove_stop_words(row['title'], stop_words)

    return df, stop_words
