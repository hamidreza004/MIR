from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


class RandomForest:
    def __init__(self):
        self.clf = None
        self.v = None
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def train(self, train_targets, train_tfIdf):
        v = DictVectorizer(sparse=False)
        X = v.fit_transform(train_tfIdf)
        self.v = v
        y = train_targets
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        self.clf.fit(X, y)

    def predict(self, doc_tfIdf):
        valid_terms = {}
        for term in doc_tfIdf.keys():
            if term in self.v.get_feature_names():
                valid_terms[term] = doc_tfIdf.get(term)
        x = self.v.fit_transform(valid_terms)[0]
        return self.clf.predict([x])

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


target_ = [1, -1]
tfIdf = [{"good": 4, "bad": 1, "girl": 2, "boy": 2}, {"good": 1, "bad": 4, "girl": 5, "boy": 2}]

rf = RandomForest()
rf.train(target_, tfIdf)
print(rf.predict({"bad": 1, "girl": 5, "good": 1, "boy": 2}))
