import bisect

positional = {}
bigram = {}
number_in_docs = {}
doc_is_available = []


def add_list_sorted(list, id, pos):
    if len(list) == 0:
        list.append([id, pos])
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
        bisect.insort(list[left], pos)
    elif right == len(list):
        list.insert(right, [id, pos])


def add_to_indexes(id, tokens, is_title):
    for ind, token in enumerate(tokens):
        if is_title:
            position = ind * 2
        else:
            position = ind * 2 + 1
        if not token in positional:
            positional[token] = []
        add_list_sorted(positional[token], id, position)


def add_single_document(description, title):
    doc_is_available.append(True)
    id = len(doc_is_available)
    add_to_indexes(id, description, is_title=False)
    add_to_indexes(id, title, is_title=True)


def add_multiple_documents(df):
    for _, row in df.iterrows():
        add_single_document(row['description'], row['title'])
