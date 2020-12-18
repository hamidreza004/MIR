import json
from compressor.gamma_code import GammaCodeDecompressor
from compressor.variable_byte import VariableByteDecompressor


class FileReader:
    def __init__(self):
        self.stop_words = []
        self.doc_is_available = []
        self.normalized_docs = {}
        self.all_tokens = []
        self.token_map = dict()
        self.bigram = dict()
        self.positional = dict()
        self.label = dict()
        self.path = "IR_files/"

    def read(self, compress_type):
        self.read_stop_words()
        self.read_doc_is_available()
        self.read_normalized_docs()
        self.read_all_tokens()
        self.read_label()
        self.read_bigram(compress_type)
        self.read_positional(compress_type)

    def read_stop_words(self):
        file = open(self.path + "stop_words.txt", "r")
        self.stop_words = json.loads(file.read())
        file.close()

    def read_doc_is_available(self):
        file = open(self.path + "doc_is_available.txt", "r")
        self.doc_is_available = json.loads(file.read())
        file.close()

    def read_normalized_docs(self):
        file = open(self.path + "normalized_docs.txt", "r")
        self.normalized_docs = json.loads(file.read())
        self.normalized_docs = {int(k): v for k, v in self.normalized_docs.items()}
        file.close()

    def read_label(self):
        file = open(self.path + "label.txt", "r")
        self.label = json.loads(file.read())
        self.label = {int(k): v for k, v in self.label.items()}
        file.close()

    def read_all_tokens(self):
        file = open(self.path + "all_tokens.txt", "r")
        self.all_tokens = json.loads(file.read())
        for i, t in enumerate(self.all_tokens):
            self.token_map[t] = i
        file.close()

    def read_bigram(self, compress_type):
        file = open(self.path + "bigram.txt", "r")
        self.bigram = self.get_decompressed_bigram(json.loads(file.read()), compress_type)
        file.close()

    def read_positional(self, compress_type):
        file = open(self.path + "positional.txt", "r")
        self.positional = self.get_decompressed_positional(json.loads(file.read()), compress_type)
        self.positional = {int(k): v for k, v in self.positional.items()}
        file.close()

    def get_decompressed_bigram(self, compressed, compress_type):
        if compress_type == "none":
            return compressed
        decompressed = dict()
        if compress_type == "gamma_code":
            decompressor = GammaCodeDecompressor()
        elif compress_type == "variable_byte":
            decompressor = VariableByteDecompressor()
        for term, posting in compressed.items():
            decompressed[term] = decompressor.get_decompressed(posting, is_positional=False)
        return decompressed

    def get_decompressed_positional(self, compressed, compress_type):
        if compress_type == "none":
            return compressed
        decompressed = dict()
        if compress_type == "gamma_code":
            decompressor = GammaCodeDecompressor()
        elif compress_type == "variable_byte":
            decompressor = VariableByteDecompressor()
        for term, posting in compressed.items():
            decompressed[term] = decompressor.get_decompressed(posting)
        return decompressed
