import json


class FileReader:
    def __init__(self, path_to_file):
        self.path = path_to_file
        self.inverted_index = ""

    def read(self):
        file = open(self.path, "r")
        self.inverted_index = json.loads(file.read())
        file.close()


# file_reader = FileReader("./test.txt")
# file_reader.read()
# print(file_reader.inverted_index)
