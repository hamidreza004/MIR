import json
import io
import sys
from compressor.gamma_code import GammaCodeCompressor
from compressor.variable_byte import VariableByteCompressor


class FileWriter:
    def __init__(self, doc_is_available, normalized_docs, all_tokens, bigram, positional):
        self.doc_is_available = doc_is_available
        self.normalized_docs = normalized_docs
        self.all_tokens = all_tokens
        self.bigram = bigram
        self.positional = positional
        self.path = "../IR_files/"

    def write(self, compress_type):
        self.write_doc_is_available()
        self.write_normailzed_docs()
        self.write_all_tokens()
        self.write_bigram(compress_type)
        self.write_positional(compress_type)

    def write_doc_is_available(self):
        file = open(self.path + "doc_is_available.txt", "w")
        file.write(json.dumps(self.doc_is_available))
        file.close()

    def write_normailzed_docs(self):
        file = open(self.path + "normalized_docs.txt", "w")
        file.write(json.dumps(self.normalized_docs))
        file.close()

    def write_all_tokens(self):
        file = open(self.path + "all_tokens.txt", "w")
        file.write(json.dumps(self.all_tokens))
        file.close()

    def write_bigram(self, compress_type):
        file = open(self.path + "bigram.txt", "w")
        file.write(json.dumps(self.get_compressed_bigram(compress_type)))
        file.close()

    def write_positional(self, compress_type):
        file = open(self.path + "positional.txt", "w", encoding="utf-8")
        compressed_positional = self.get_compressed_positional(compress_type)
        json.dump(compressed_positional, file, ensure_ascii=False)
        #file.write(json.dumps(compressed_positional))
        file.close()

    def get_compressed_positional(self, compress_type):
        if compress_type == "none":
            return self.positional
        compressed = dict()
        if compress_type == "gamma_code":
            compressor = VariableByteCompressor()
        elif compress_type == "variable_byte":
            compressor = GammaCodeCompressor()
        for term, posting in self.positional.items():
            compressed[term] = compressor.get_compressed(posting)
        return compressed

    def get_compressed_bigram(self, compress_type):
        if compress_type == "none":
            return self.bigram
        compressed = dict()
        if compress_type == "gamma_code":
            compressor = VariableByteCompressor()
        elif compress_type == "variable_byte":
            compressor = GammaCodeCompressor()
        for term, posting in self.bigram.items():
            compressed[term] = compressor.get_compressed(posting, is_positional=False)
        return compressed


inv_index = dict()
inv_index["salam"] = [[0, [3, 4, 6]], [10, [3, 7, 24]], [15, [23, 34523452345]]]
# inv_index["boos"] = [[7, [4, 6]], [10, [0, 24]], [11, [234, 345]]]

file_writer = FileWriter(None, None, None, None, inv_index)
file_writer.write_positional("gamma_code")
