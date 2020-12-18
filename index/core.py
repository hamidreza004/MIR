import bisect
from collections import Counter
from preprocess.TF_IDF import create_tf_idf
import math

# Basic state:
positional = {}
bigram = {}
all_tokens = []
doc_is_available = [True]
normalize_doc = {}
label = {}
# Can reproduce:
token_map = {}


def token_exists(token):
    return token in token_map


def get_token_id(token):
    if not token in token_map:
        token_map[token] = len(all_tokens)
        all_tokens.append(token)
    return token_map[token]


def add_list_sorted(my_list, id, pos):
    if len(my_list) == 0:
        my_list.append([id, [pos]])
        return
    left = 0
    right = len(my_list)
    while left < right - 1:
        mid = (left + right) // 2
        if my_list[mid][0] > id:
            right = mid
        else:
            left = mid
    if my_list[left][0] == id:
        bisect.insort(my_list[left][1], pos)
    else:
        my_list.insert(right, [id, [pos]])


def add_to_indexes(id, tokens, is_title):
    for ind, token in enumerate(tokens):
        token_id = get_token_id(token)
        if is_title:
            position = ind * 2
        else:
            position = ind * 2 + 1
        if not token_id in positional:
            positional[token_id] = []
        add_list_sorted(positional[token_id], id, position)
        new_token = "$" + token + "$"
        for i in range(len(new_token) - 1):
            new_str = new_token[i] + new_token[i + 1]
            if not new_str in bigram:
                bigram[new_str] = []
            loc = bisect.bisect_left(bigram[new_str], token_id, lo=0, hi=len(bigram[new_str]))
            if loc >= len(bigram[new_str]) or bigram[new_str][loc] != token_id:
                bisect.insort(bigram[new_str], token_id)


def add_single_document(description, title, classifier=None):
    id = len(doc_is_available)
    doc_is_available.append(True)
    add_to_indexes(id, description, is_title=False)
    add_to_indexes(id, title, is_title=True)
    merged_tokens = description[:]
    merged_tokens.extend(title)
    normalize_doc[id] = sum([c * c for term, c in Counter(merged_tokens).items()])

    if classifier is not None:
        N = len(doc_is_available)
        term_freq = {}
        for word in merged_tokens:
            if word not in term_freq.keys():
                term_freq[word] = 1
            else:
                term_freq[word] += 1
        tf_idf_doc = {}
        for word in merged_tokens:
            tf_idf_doc[word] = term_freq[word] * math.log(N / len(positional[token_map[word]]))
        label[id] = classifier.predict(tf_idf_doc)
    return id


def remove_document(id):
    doc_is_available[id] = False


def add_multiple_documents(df, classifier):
    for _, row in df.iterrows():
        add_single_document(row['description'], row['title'])
    df['text'] = df['description'] + df['title']
    df = df[['text']]
    vocab, tf_idf = create_tf_idf(df)
    counter = 1
    for doc in tf_idf:
        label[counter] = classifier.predict(doc)
        counter += 1
