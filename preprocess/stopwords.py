def remove_stop_words(tokens, stop_words):
    return [word for word in tokens if not word in stop_words[0]]


def find_stop_words(all_tokens_dic, stop_word_ratio):
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
