import bisect

top_jaccard_items_candidate = 10


def edit_distance(str1, str2):
    dis = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
    for i in range(len(str1) + 1):
        for j in range(len(str2) + 1):
            if min(i, j) == 0:
                dis[i][j] = max(i, j)
            else:
                dis[i][j] = min(min(dis[i - 1][j], dis[i][j - 1]) + 1,
                                dis[i - 1][j - 1] + (1 if str1[i - 1] != str2[j - 1] else 0))
    return dis[len(str1)][len(str2)]


def get_token_bi_words(token):
    token = "$" + token + "$"
    return sorted(set([token[i] + token[i + 1] for i in range(len(token) - 1)]))


def correct(query, index, stop_words, lang):
    cleaned_query = lang.clean_raw(query)
    corrected_query = []
    edit_distance_value = 0
    jaccard_distance_value = 0
    for token in cleaned_query:
        if not token in index.all_tokens and not token in stop_words:
            bi_words = get_token_bi_words(token)
            jaccard_candidates = []
            for bi_word in bi_words:
                if bi_word in index.bigram:
                    for token_id in index.bigram[bi_word]:
                        intersection = 0
                        sum_of_subset_sizes = len(bi_words) + len(get_token_bi_words(index.all_tokens[token_id]))
                        for bi_word_2 in bi_words:
                            if bi_word_2 in index.bigram:
                                idx = bisect.bisect_left(index.bigram[bi_word_2], token_id)
                                if idx < len(index.bigram[bi_word_2]) and index.bigram[bi_word_2][idx] == token_id:
                                    intersection += 1
                        jaccard_candidates.append((1.0 - intersection / (sum_of_subset_sizes - intersection), token_id))
            jaccard_candidates = sorted(jaccard_candidates)[:top_jaccard_items_candidate]
            edit_distance_candidates = []
            for candidate in jaccard_candidates:
                org_str = index.all_tokens[candidate[1]]
                edit_distance_candidates.append((edit_distance(token, org_str), -len(org_str), org_str, candidate[0]))
            e, _, corrected_token, j = sorted(edit_distance_candidates)[0]
            edit_distance_value += e
            corrected_query.append(corrected_token)
            jaccard_distance_value += j
        else:
            corrected_query.append(token)
    return ' '.join(cleaned_query), ' '.join(corrected_query), jaccard_distance_value, edit_distance_value
