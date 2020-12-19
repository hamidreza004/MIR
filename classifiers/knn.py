from math import sqrt
from random import sample


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
        self.target1_predicted1 = 0
        self.target2_predicted2 = 0
        self.target2_predicted1 = 0
        self.target1_predicted2 = 0

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

        sample_size = int(len(train_tfIdf) * 0.3)
        sample_indices = sample(range(0, len(train_tfIdf)), sample_size)
        validation_size = int(sample_size * 0.1)

        validation_data = []
        validation_target = []
        for index in sample_indices[0:validation_size]:
            validation_data.append(train_tfIdf[index])
            validation_target.append(train_targets[index])

        train_data = []
        for index in sample_indices[validation_size: sample_size]:
            train_data.append(data[index])

        self.data = train_data

        K_VALUES = [1, 5, 9]

        knn = KNN()
        knn.data = train_data
        knn.vocab = train_vocab
        arg_max = 0
        max_acc = 0

        print("")
        for k in K_VALUES:
            knn.k = k
            knn.test(validation_target, validation_data)
            acc = knn.get_accuracy()
            print("Accuracy for " + str(k) + "-NN: " + str(acc))
            if acc >= max_acc:
                max_acc = acc
                arg_max = k

        print("Selected K for K-NN is " + str(arg_max) + " with accuracy of " + str(max_acc))
        print("")
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
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    def test(self, test_targets, test_tfIdf):
        self.target1_predicted1 = 0
        self.target2_predicted2 = 0
        self.target2_predicted1 = 0
        self.target1_predicted2 = 0

        predicted = []
        for i, doc_tfIdf in enumerate(test_tfIdf):
            predicted.append(self.predict(doc_tfIdf))
            if test_targets[i] == 1:
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
