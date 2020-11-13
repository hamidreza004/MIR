from collections import Counter
from math import log, sqrt

number_of_top_results = 20


def search(token_ids, index, documents=None):
    N = len(index.doc_is_available)
    query_terms = Counter(token_ids).items()
    normalize_query = sum([c * c for _, c in query_terms])
    candidates = {}
    for token_id, count in query_terms:
        weight_term_query = (1 + log(count)) * log(N / len(index.positional[token_id])) / sqrt(normalize_query)
        for doc_pos in index.positional[token_id]:
            tf_document = len(doc_pos) - 1
            document = doc_pos[0]
            if documents is not None and document not in documents:
                continue
            weight_term_document = (1 + log(tf_document)) * 1 / sqrt(index.normalize_doc[document])
            similarity = weight_term_query * weight_term_document
            if not document in candidates:
                candidates[document] = 0
            candidates[document] -= similarity
    results = []
    idx = 0
    for document, score in sorted(candidates.items(), key=lambda item: item[1]):
        if idx >= number_of_top_results:
            break
        if not index.doc_is_available[document]:
            continue
        results.append((document, -score))
        idx += 1
    return results
