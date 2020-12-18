from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer


class SVM:
    def __init__(self):
        self.clf = None
        self.v = None
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def train(self, train_targets, train_tfIdf, C=1.0):
        v = DictVectorizer(sparse=False)
        X = v.fit_transform(train_tfIdf)
        self.v = v
        y = train_targets
        self.clf = SVC(C=C)
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

#
# target_ = [1, -1]
# tfIdf = [{"good": 4, "bad": 0, "girl": 2, "boy": 2}, {"good": 1, "bad": 4, "boy": 20}]
#
#
# X_train = []
# y_train = []
#
# validation_data = X_train[:int(len(X_train) * 0.1)]
# validation_target = y_train[:int(len(y_train) * 0.1)]
# train_data = X_train[int(len(X_train) * 0.1):]
# train_target = y_train[int(len(y_train) * 0.1):]
#
# C_VALUES = [0.5, 1, 1.5, 2]
#
# svm = SVM()
# arg_max = 0
# max_acc = 0
#
# for c in C_VALUES:
#     svm.train(train_target, train_data, C=c)
#     svm.test(validation_target, validation_data)
#     acc = svm.get_accuracy()
#     if acc > max_acc:
#         max_acc = acc
#         arg_max = c
