import search.LNC_LTC as LNC_LTC


def is_document_with_window_terms(doc, max_pos, token_ids, index, window_size, title=False):
    for i in range(max_pos + 1):
        flg = True
        for token_id in token_ids:
            left = 0
            right = len(index.positional[token_id])
            while left < right - 1:
                mid = (left + right) // 2
                if index.positional[token_id][mid][0] > doc:
                    right = mid
                else:
                    left = mid
            if index.positional[token_id][left][0] == doc:
                find_pos = False
                for pos in index.positional[token_id][left][1]:
                    if (1 - pos % 2) == title and pos // 2 - i + 1 <= window_size and pos // 2 >= i:
                        find_pos = True
                        break
                if not find_pos:
                    flg = False
                    break
            else:
                flg = False
                break
        if flg:
            return True
    return False


def search(token_ids, index, window_size, documents=None):
    title_docs = []
    desc_docs = []
    for tmp_doc_pos in index.positional[token_ids[0]]:
        doc = tmp_doc_pos[0]
        if documents is not None and doc not in documents:
            continue
        max_pos = tmp_doc_pos[1][-1] // 2 + 1
        if is_document_with_window_terms(doc, max_pos, token_ids, index, window_size, True):
            title_docs.append(doc)
        if is_document_with_window_terms(doc, max_pos, token_ids, index, window_size, False):
            desc_docs.append(doc)
    title_docs = LNC_LTC.search(token_ids, index, title_docs)
    desc_docs = LNC_LTC.search(token_ids, index, desc_docs)
    return title_docs, desc_docs
