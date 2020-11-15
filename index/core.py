import bisect
from collections import Counter

# Basic state:
positional = {}
bigram = {}
all_tokens = []
doc_is_available = [True]
normalize_doc = {}
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


def add_single_document(description, title):
    id = len(doc_is_available)
    doc_is_available.append(True)
    add_to_indexes(id, description, is_title=False)
    add_to_indexes(id, title, is_title=True)
    merged_tokens = description[:]
    merged_tokens.extend(title)
    normalize_doc[id] = sum([c * c for term, c in Counter(merged_tokens).items()])
    return id


def remove_document(id):
    doc_is_available[id] = False


def add_multiple_documents(df):
    for _, row in df.iterrows():
        add_single_document(row['description'], row['title'])
