import bisect

positional = {}
bigram = {}
number_in_docs = {}
token_map = {}
all_tokens = []
doc_is_available = []


def token_exists(token):
    return token in token_map


def get_token_id(token):
    if not token in token_map:
        token_map[token] = len(all_tokens)
        all_tokens.append(token)
    return token_map[token]


def add_list_sorted(list, id, pos):
    if len(list) == 0:
        list.append([id, [pos]])
        return
    left = 0
    right = len(list)
    while left < right - 1:
        mid = (left + right) // 2
        if list[mid][0] > id:
            right = mid
        else:
            left = mid
    if list[left][0] == id:
        bisect.insort(list[left][1], pos)
    elif right == len(list):
        list.insert(right, [id, [pos]])


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
        i = 0
        for i in range(len(token) - 1):
            new_str = token[i] + token[i + 1]
            if not new_str in bigram:
                bigram[new_str] = []
            loc = bisect.bisect_left(bigram[new_str], token_id, lo=0, hi=len(bigram[new_str]))
            if loc >= len(bigram[new_str]) or bigram[new_str][loc] != token_id:
                bisect.insort(bigram[new_str], token_id)


def add_single_document(description, title):
    doc_is_available.append(True)
    id = len(doc_is_available)
    add_to_indexes(id, description, is_title=False)
    add_to_indexes(id, title, is_title=True)


def add_multiple_documents(df):
    for _, row in df.iterrows():
        add_single_document(row['description'], row['title'])
