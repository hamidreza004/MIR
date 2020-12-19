from math import sqrt


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row2) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


class KNN:
    def __init__(self):
        self.vocab = []
        self.data = []
        self.k = 1
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def get_neighbors(self, test_row):
        distances = []
        for train_row in self.data:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    def train(self, train_targets, train_vocab, train_tfIdf):
        self.vocab = train_vocab

        data = []
        for i, doc in enumerate(train_tfIdf):
            x = []
            for term in train_vocab:
                if term in doc.keys():
                    x.append(doc.get(term))
                else:
                    x.append(0)

            x.append(train_targets[i])
            data.append(x)

        validation_data = train_tfIdf[:int(len(train_tfIdf) * 0.04)]
        validation_target = train_targets[:int(len(train_targets) * 0.04)]

        train_data = data[int(len(data) * 0.04):int(len(data) * 0.4)]

        self.data = train_data

        K_VALUES = [1, 5, 9]

        knn = KNN()
        knn.data = train_data
        knn.vocab = train_vocab
        arg_max = 0
        max_acc = 0

        # print("done pre")

        for k in K_VALUES:
            knn.k = k
            knn.test(validation_target, validation_data)
            acc = knn.get_accuracy()
            print("knn k / acc", k, acc)
            if acc > max_acc:
                max_acc = acc
                arg_max = k

        print("knn arg_max / max_acc", arg_max, max_acc)
        self.k = arg_max

    def predict(self, doc_tfIdf):
        x = []
        for term in self.vocab:
            if term in doc_tfIdf.keys():
                x.append(doc_tfIdf.get(term))
            else:
                x.append(0)

        neighbors = self.get_neighbors(x)
        output_values = [row[-1] for row in neighbors]
        # print(output_values)
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    def test(self, test_targets, test_tfIdf):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        predicted = []
        print(len(test_targets))
        for i, doc_tfIdf in enumerate(test_tfIdf):
            predicted.append(self.predict(doc_tfIdf))
            # print("predicted")
            if test_targets[i] == 1:
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


# target_ = [1, -1]
# vocab = ["good", "bad", "girl", "boy"]
# tfIdf = [{"good": 4, "bad": 1, "girl": 2, "boy": 2}, {"good": 1, "bad": 4, "girl": 5, "boy": 2}]

