import math


class NaiveBayes:
    def __init__(self):
        self.prior = {}
        self.likelihood = {1: {}, -1: {}}
        self.total = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

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
                    p_c += document_tfIdf.get(term) * self.likelihood.get(c).get(term)
                else:
                    p_c += 1 / self.total[c]
            p[c] = p_c
        if p.get(1) >= p.get(-1):
            return 1
        else:
            return -1

    def test(self, test_targets, test_tfIdf):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        predicted = []
        for doc_tfIdf in test_tfIdf:
            predicted.append(self.predict(doc_tfIdf))
        for i, target in enumerate(test_targets):
            if target == 1:
                if predicted[i] == 1:
                    self.tp += 1
                else:
                    self.fn += 1
            else:
                if predicted[i] == 1:
                    self.fp += 1
                else:
                    self.tn += 1

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp / (self.tp + self.fn)

    def get_F1(self):
        P = self.get_precision()
        R = self.get_recall()
        return (2 * P * R) / (P + R)


# good boy good boy good girl good girl bad             --> 1
# bad bad boy boy girl bad bad good girl girl girl girl --> -1
# good girl girl girl girl boy bad good good good       --> 1

target_ = [1, -1]
vocab = ["good", "bad", "girl", "boy"]
tfIdf = [{"good": 4, "bad": 1, "boy": 2}, {"good": 1, "bad": 4, "girl": 5, "boy": 2}]

nb = NaiveBayes()
nb.train(target_, vocab, tfIdf)
print(nb.prior)
print(nb.likelihood)
nb.test(target_, tfIdf)
print(nb.get_F1())
print(nb.predict({"good": 1, "bad": 4, "girl": 5, "hosr": 35}))
