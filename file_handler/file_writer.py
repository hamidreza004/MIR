import json
import os
from compressor.gamma_code import GammaCodeCompressor
from compressor.variable_byte import VariableByteCompressor


class FileWriter:
    def __init__(self, stop_words, doc_is_available, normalized_docs, all_tokens, bigram, positional, label):
        self.stop_words = stop_words
        self.doc_is_available = doc_is_available
        self.normalized_docs = normalized_docs
        self.all_tokens = all_tokens
        self.bigram = bigram
        self.positional = positional
        self.label = label
        self.path = "IR_files/"
        try:
            os.mkdir(self.path)
        except:
            pass

    def write(self, compress_type):
        self.write_stop_words()
        self.write_doc_is_available()
        self.write_normailzed_docs()
        self.write_all_tokens()
        self.write_positional_none()
        self.write_bigram_none()
        self.write_label()
        self.write_bigram(compress_type)
        self.write_positional(compress_type)

    def write_stop_words(self):
        file = open(self.path + "stop_words.txt", "w")
        file.write(json.dumps(self.stop_words))
        file.close()

    def write_doc_is_available(self):
        file = open(self.path + "doc_is_available.txt", "w")
        file.write(json.dumps(self.doc_is_available))
        file.close()

    def write_normailzed_docs(self):
        file = open(self.path + "normalized_docs.txt", "w")
        file.write(json.dumps(self.normalized_docs))
        file.close()

    def write_label(self):
        file = open(self.path + "label.txt", "w")
        file.write(json.dumps(self.label))
        file.close()

    def write_all_tokens(self):
        file = open(self.path + "all_tokens.txt", "w")
        file.write(json.dumps(self.all_tokens))
        file.close()

    def write_bigram(self, compress_type):
        file = open(self.path + "bigram.txt", "w")
        compressed_bigram = self.get_compressed_bigram(compress_type)
        json_string = json.dumps(compressed_bigram, ensure_ascii=False).encode('utf8')
        file.write(json_string.decode())
        file.close()

    def write_bigram_none(self):
        file = open(self.path + "bigram_none.txt", "w")
        compressed_bigram = self.get_compressed_bigram("none")
        json_string = json.dumps(compressed_bigram, ensure_ascii=False).encode('utf8')
        file.write(json_string.decode())
        file.close()

    def get_bigram_size(self):
        return os.stat(self.path + "bigram_none.txt").st_size, os.stat(self.path + "bigram.txt").st_size

    def write_positional(self, compress_type):
        file = open(self.path + "positional.txt", "w", encoding="utf-8")
        compressed_positional = self.get_compressed_positional(compress_type)
        json_string = json.dumps(compressed_positional, ensure_ascii=False).encode('utf8')
        file.write(json_string.decode())
        file.close()

    def write_positional_none(self):
        file = open(self.path + "positional_none.txt", "w", encoding="utf-8")
        compressed_positional = self.get_compressed_positional("none")
        json_string = json.dumps(compressed_positional, ensure_ascii=False).encode('utf8')
        file.write(json_string.decode())
        file.close()

    def get_positional_size(self):
        return os.stat(self.path + "positional_none.txt").st_size, os.stat(self.path + "positional.txt").st_size

    def get_compressed_positional(self, compress_type):
        if compress_type == "none":
            return self.positional
        compressed = dict()
        if compress_type == "gamma_code":
            compressor = GammaCodeCompressor()
        elif compress_type == "variable_byte":
            compressor = VariableByteCompressor()
        for term, posting in self.positional.items():
            compressed[term] = compressor.get_compressed(posting)
        return compressed

    def get_compressed_bigram(self, compress_type):
        if compress_type == "none":
            return self.bigram
        compressed = dict()
        if compress_type == "gamma_code":
            compressor = GammaCodeCompressor()
        elif compress_type == "variable_byte":
            compressor = VariableByteCompressor()
        for term, posting in self.bigram.items():
            compressed[term] = compressor.get_compressed(posting, is_positional=False)
        return compressed

#
# inv_index = dict()
# inv_index["salam"] = [[0, [3, 4, 6]], [10, [3, 7, 24]], [15, [23, 35345]]]
# inv_index["boos"] = [[7, [4, 6]], [10, [0, 24]], [11, [234, 345]]]
#
# file_writer = FileWriter(None, None, None, None, None, inv_index)
# file_writer.write_positional("none")
# file_writer.write_positional_none()
# print(file_writer.get_positional_size())
