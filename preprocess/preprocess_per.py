import hazm
from preprocess.stopwords import remove_stop_words, find_stop_words

normalizer = hazm.Normalizer()
stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
stop_word_ratio = 0.002
stop_words = []


def clean_raw(raw):
    raw = normalizer.normalize(raw)
    tokens = hazm.word_tokenize(raw)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
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
