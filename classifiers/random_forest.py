from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.clf = None
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def train(self, train_targets, train_vocab, train_tfIdf):
        X = []
        for doc in train_tfIdf:
            x = []
            for term in train_vocab:
                x.append(doc.get(term))
            X.append(x)
        y = train_targets
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        self.clf.fit(X, y)

    def predict(self, doc_tfIdf):
        return self.clf.predict([doc_tfIdf])


target_ = [1, -1]
vocab = ["good", "bad", "girl", "boy"]
tfIdf = [{"good": 4, "bad": 1, "girl": 2, "boy": 2}, {"good": 1, "bad": 4, "girl": 5, "boy": 2}]

rf = RandomForest()
rf.train(target_, vocab, tfIdf)
print(rf.predict([1, 4, 5, 2]))
