from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self):
        self.clf = None
        self.vocab = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def train(self, train_targets, train_vocab, train_tfIdf):
        self.vocab = train_vocab
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
        x = []
        for term in self.vocab:
            if term in doc_tfIdf.keys():
                x.append(doc_tfIdf.get(term))
            else:
                x.append(0)
        return self.clf.predict([x])


target_ = [1, -1]
vocab = ["good", "bad", "girl", "boy"]
tfIdf = [{"good": 4, "bad": 1, "girl": 2, "boy": 2}, {"good": 1, "bad": 4, "girl": 5, "boy": 2}]

rf = RandomForest()
rf.train(target_, vocab, tfIdf)
print(rf.predict({"bad": 1, "girl": 2, "good": 4, "boy": 2}))
