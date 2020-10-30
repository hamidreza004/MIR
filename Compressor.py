from bitstring import BitArray


class GammaCodeCompressor:

    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
        self.bit_stream = ""
        self.padding = 0
        self.last_document = 0

    def create_compressed_file(self, positional_posting_list):
        self.compress(positional_posting_list)
        self.set_padding()
        self.write_to_file()

    def compress(self, positional_posting_list):
        self.bit_stream = ""
        for i in range(0, len(positional_posting_list)):
            if i % 2 == 0:
                # compress document number
                document = positional_posting_list[i]
                self.bit_stream += self.get_gamma_code(document - self.last_document)
                self.last_document = document
            else:
                # compress document number
                positions = positional_posting_list[i]
                self.bit_stream += self.get_gamma_code(len(positions))
                last_position = 0
                for position in positions:
                    self.bit_stream += self.get_gamma_code(position - last_position)
                    last_position = position

    def set_padding(self):
        self.padding = (8 - ((len(self.bit_stream) + 3) % 8) % 8)
        for i in range(0, self.padding):
            self.bit_stream += '1'
        self.bit_stream = self.to_3bit_binary(self.padding) + self.bit_stream

    def get_unary_code(self, decimal_number):
        unary_code = ""
        for i in range(0, decimal_number):
            unary_code += '1'
        unary_code += '0'
        return unary_code

    def get_binary_code(self, deciaml_number):
        return bin(deciaml_number)[2:]

    def get_gamma_code(self, decimal_number):
        offset = self.get_binary_code(decimal_number)[1:]
        length = self.get_unary_code(len(offset))
        return length + offset

    def to_3bit_binary(self, decimal_number):
        num = decimal_number
        reversed_binary = ""
        for i in range(0, 3):
            if num % 2 == 1:
                reversed_binary += '1'
            else:
                reversed_binary += '0'
            num = num // 2
        binary = ""
        for i in reversed(reversed_binary):
            binary += i
        return binary

    def write_to_file(self):
        byte_stream = ""
        i = 0
        while i < len(self.bit_stream):
            byte_stream += chr(int(self.bit_stream[i: i + 8], 2))
            i += 8
        file = open(self.path_to_file, "w")
        file.write(byte_stream)
        file.close()


class GammaCodeDecompressor:

    def __init__(self, file_location):
        self.path_to_file = file_location
        self.bit_stream = ''
        self.read_file_in_binary()
        self.iterator = 3
        self.bit_stream_len = len(self.bit_stream)

    def read_file_in_binary(self):
        file = open(self.path_to_file, "r")
        byte = file.read(1)
        while byte:
            self.bit_stream += format(ord(byte), '08b')
            byte = file.read(1)
        file.close()

    def get_positional_posting_list(self):
        posting_list = []
        document_number = 0
        document_diff = self.get_next_number()
        while document_diff is not None:
            document_number += document_diff
            posting_list.append(document_number)
            positions_len = self.get_next_number()
            positions = []
            position = 0
            for i in range(0, positions_len):
                position += self.get_next_number()
                positions.append(position)
            posting_list.append(positions)
            document_diff = self.get_next_number()
        return posting_list

    def get_next_number(self):
        length = 0
        while self.bit_stream[self.iterator] == '1':
            length += 1
            self.iterator += 1
            if self.iterator == self.bit_stream_len:
                return None
        self.iterator += 1
        offset = self.bit_stream[self.iterator: self.iterator + length]
        self.iterator += length
        return int('1' + offset, 2)


gammaCodeCompressor = GammaCodeCompressor("./ted_postings_positional_gamma")
postings = [10, [1, 4, 5], 15, [2, 6, 8]]
gammaCodeCompressor.create_compressed_file(postings)
print(gammaCodeCompressor.bit_stream)

unzip = GammaCodeDecompressor("./ted_postings_positional_gamma")
print(unzip.bit_stream)
print(unzip.get_positional_posting_list())
