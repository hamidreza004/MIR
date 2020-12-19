from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


class RandomForest:
    def __init__(self):
        self.clf = None
        self.v = None
        self.target1_predicted1 = 0
        self.target2_predicted2 = 0
        self.target2_predicted1 = 0
        self.target1_predicted2 = 0

    def train(self, train_targets, train_tfIdf):
        v = DictVectorizer(sparse=False)
        X = v.fit_transform(train_tfIdf)
        self.v = v
        y = train_targets
        self.clf = RandomForestClassifier(max_depth=30, random_state=0)
        self.clf.fit(X, y)

    def predict(self, doc_tfIdf):
        x = []
        for term in self.v.get_feature_names():
            if term in doc_tfIdf:
                x.append(doc_tfIdf.get(term))
            else:
                x.append(0)
        return int(self.clf.predict([x])[0])

    def test(self, test_targets, test_tfIdf):
        self.target1_predicted1 = 0
        self.target2_predicted2 = 0
        self.target2_predicted1 = 0
        self.target1_predicted2 = 0

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
