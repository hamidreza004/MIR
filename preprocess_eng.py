import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer

snowball = SnowballStemmer("english")
lemma = WordNetLemmatizer()
stop_word_ratio = 0.0026


def clean_raw(raw):
    tokens = word_tokenize(raw.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [snowball.stem(word) for word in tokens]
    tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
    tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
    return tokens


def remove_stop_words(tokens, stop_words):
    return [word for word in tokens if not word in stop_words[0]]


def find_stop_words(all_tokens_dic):
    stop_words = [[], []]
    stop_point = len(all_tokens_dic.items()) * stop_word_ratio
    i = 0
    for token, count in sorted(all_tokens_dic.items(), key=lambda item: item[1], reverse=True):
        i += 1
        stop_words[0].append(token)
        stop_words[1].append(count)
        if i > stop_point:
            break
    return stop_words


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

    stop_words = find_stop_words(all_tokens_dic)

    for index, row in df.iterrows():
        row['description'] = remove_stop_words(row['description'], stop_words)
        row['title'] = remove_stop_words(row['title'], stop_words)

    return df, stop_words
