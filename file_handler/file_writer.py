import json


class FileWriter:
    def __init__(self, path_to_file):
        self.path = path_to_file

    def write(self, inverted_index):
        file = open(self.path, "w")
        file.write(json.dumps(inverted_index))
        file.close()


# inv_index = dict()
# inv_index["salam"] = [0, [3, 4, 6], 10, [3, 7, 24], 15, [23, 34523452345]]
# inv_index["boos"] = [7, [4, 6], 10, [0, 24], 11, [234, 345]]
#
# file_writer = FileWriter("./test.txt")
# file_writer.write(inv_index)
