import math


class NaiveBayes:
    def __init__(self):
        self.prior = {}
        self.likelihood = {1: {}, -1: {}}
        self.total = []
        self.target1_predicted1 = 0
        self.target2_predicted2 = 0
        self.target2_predicted1 = 0
        self.target1_predicted2 = 0

    def train(self, train_targets, train_vocab, train_tfIdf):
        self.prior = {}
        self.likelihood = {1: {}, -1: {}}
        self.total = []

        # likelihood
        t = {1: {}, -1: {}}
        total = {1: len(train_vocab), -1: len(train_vocab)}
        for term in train_vocab:
            count_term_c1 = 0
            count_term_c2 = 0
            for doc_id, target in enumerate(train_targets):
                if target == 1:
                    if term in train_tfIdf[doc_id]:
                        count_term_c1 += train_tfIdf[doc_id].get(term)
                elif target == -1:
                    if term in train_tfIdf[doc_id]:
                        count_term_c2 += train_tfIdf[doc_id].get(term)
            t[1][term] = count_term_c1
            total[1] += count_term_c1
            t[-1][term] = count_term_c2
            total[-1] += count_term_c2
        self.total = total
        for c in t.keys():
            for term in train_vocab:
                self.likelihood[c][term] = (t[c][term] + 1) / total[c]

        # prior
        count_c1 = 0
        count_c2 = 0
        for target in train_targets:
            if target == 1:
                count_c1 += 1
            elif target == -1:
                count_c2 += 1
        self.prior[1] = count_c1 / len(train_targets)
        self.prior[-1] = count_c2 / len(train_targets)

    def predict(self, document_tfIdf):
        p = {}
        for c in [1, -1]:
            p_c = math.log(self.prior.get(c))
            for term in document_tfIdf:
                if term in self.likelihood.get(c):
                    p_c += document_tfIdf.get(term) * math.log(self.likelihood.get(c).get(term))
                else:
                    p_c += 1 / self.total[c]
            p[c] = p_c
        if p.get(1) >= p.get(-1):
            return 1
        else:
            return -1

    def test(self, test_targets, test_tfIdf):
        self.target1_predicted1 = 0
        self.target1_predicted2 = 0
        self.target2_predicted2 = 0
        self.target2_predicted1 = 0

        predicted = []
        for doc_tfIdf in test_tfIdf:
            predicted.append(self.predict(doc_tfIdf))
        for i, target in enumerate(test_targets):
            if target == 1:
                if predicted[i] == 1:
                    self.target1_predicted1 += 1
                else:
                    self.target1_predicted2 += 1
            else:
                if predicted[i] == 1:
                    self.target2_predicted1 += 1
                else:
                    self.target2_predicted2 += 1

    def get_accuracy(self):
        return (self.target1_predicted1 + self.target2_predicted2) / (
                self.target1_predicted1 + self.target2_predicted2 + self.target2_predicted1 + self.target1_predicted2)

    def get_precision_c1(self):
        return self.target1_predicted1 / (self.target1_predicted1 + self.target2_predicted1)

    def get_precision_c2(self):
        return self.target2_predicted2 / (self.target2_predicted2 + self.target1_predicted2)

    def get_recall_c1(self):
        return self.target1_predicted1 / (self.target1_predicted1 + self.target1_predicted2)

    def get_recall_c2(self):
        return self.target2_predicted2 / (self.target2_predicted2 + self.target2_predicted1)

    def get_F1_c1(self):
        P = self.get_precision_c1()
        R = self.get_recall_c1()
        return (2 * P * R) / (P + R)

    def get_F1_c2(self):
        P = self.get_precision_c2()
        R = self.get_recall_c2()
        return (2 * P * R) / (P + R)
