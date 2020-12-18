import math


def create_tf_idf(df):
    document_freq = {}
    for index, row in df.iterrows():
        doc = row['text']
        mark = {}
        for word in doc:
            if word not in mark.keys():
                if word not in document_freq.keys():
                    document_freq[word] = 1
                else:
                    document_freq[word] += 1
            mark[word] = True
    tf_idf = []
    for index, row in df.iterrows():
        doc = row['text']
        term_freq = {}
        for word in doc:
            if word not in term_freq.keys():
                term_freq[word] = 1
            else:
                term_freq[word] += 1
        tf_idf_doc = {}
        for word in term_freq:
            tf_idf_doc[word] = term_freq[word] * math.log(df.shape[0] / document_freq[word])
        tf_idf.append(tf_idf_doc)
    return list(document_freq.keys()), tf_idf
