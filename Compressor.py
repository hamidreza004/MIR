class GammaCodeCompressor:

    # def __init__(self):
    #     self.bit_stream = ""
    #     self.padding = 0
    #     self.last_document = 0

    def get_compressed(self, positional_posting_list):
        compressed = self.compress(positional_posting_list)
        padded_compressed = self.set_padding(compressed)
        return self.to_byte(padded_compressed)

    # def create_compressed_file(self, positional_posting_list):
    #     self.compress(positional_posting_list)
    #     self.set_padding()
    #     self.write_to_file()

    # def add_to_compressed_file(self, positional_posting_list):
    #     first_byte, last_byte = self.get_first_last_byte()
    #     padding_number = int(first_byte[0:3], 2)
    #     remaining_data = last_byte[0: 8 - padding_number]
    #     self.compress(positional_posting_list)
    #     self.bit_stream = remaining_data + self.bit_stream
    #     self.update_padding()
    #     new_fist_byte = self.to_3bit_binary(self.padding) + first_byte[3:8]
    #     self.update_file(new_fist_byte)

    def compress(self, positional_posting_list):
        compressed = ""
        last_document_number = 0
        for i in range(0, len(positional_posting_list)):
            if i % 2 == 0:
                # compress document number
                document_number = positional_posting_list[i]
                compressed += self.get_gamma_code(document_number - last_document_number)
                last_document_number = document_number
            else:
                # compress document number
                positions = positional_posting_list[i]
                compressed += self.get_gamma_code(len(positions))
                last_position = 0
                for position in positions:
                    compressed += self.get_gamma_code(position - last_position)
                    last_position = position
        return compressed

    def set_padding(self, compressed):
        padding_amount = (8 - ((len(compressed) + 3) % 8) % 8)
        for i in range(0, padding_amount):
            compressed += '1'
        return self.to_3bit_binary(padding_amount) + compressed

    # def update_padding(self):
    #     self.padding = (8 - (len(self.bit_stream) % 8) % 8)
    #     for i in range(0, self.padding):
    #         self.bit_stream += '1'

    def write_to_file(self):
        byte_stream = ""
        i = 0
        bit_stream_len = len(self.bit_stream)
        while i < bit_stream_len:
            byte_stream += chr(int(self.bit_stream[i: i + 8], 2))
            i += 8
        file = open(self.path_to_file, "w")
        file.write(byte_stream)
        file.close()

    def to_byte(self, bit_stream):
        byte_stream = ""
        i = 0
        while i < len(bit_stream):
            byte_stream += chr(int(bit_stream[i: i + 8], 2))
            i += 8
        return byte_stream

    # def update_file(self, first_byte):
    #     byte_stream = []
    #     i = 0
    #     bit_stream_len = len(self.bit_stream)
    #     while i < bit_stream_len:
    #         byte_stream.append(int(self.bit_stream[i: i + 8], 2))
    #         i += 8
    #     file = open(self.path_to_file, 'r+b')
    #     file.write(bytearray([int(first_byte, 2)]))
    #     file.seek(-1, 2)
    #     file.write(bytearray(byte_stream))
    #     file.close()

    # def get_first_last_byte(self):
    #     file = open(self.path_to_file, 'rb')
    #     first_byte = format(ord(file.read(1)), '08b')
    #     file.seek(-1, 2)
    #     last_byte = format(ord(file.read(1)), '08b')
    #     file.close()
    #     return first_byte, last_byte

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


class GammaCodeDecompressor:

    def __init__(self, file_location):
        self.path_to_file = file_location
        self.bit_stream = ''
        self.read_file_in_binary()
        self.iterator = 3

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

    def read_file_in_binary(self):
        file = open(self.path_to_file, "r")
        byte = file.read(1)
        while byte:
            self.bit_stream += format(ord(byte), '08b')
            byte = file.read(1)
        file.close()

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


postings = [10, [1, 4, 5], 15, [2, 6, 8]]
zipper = GammaCodeCompressor()
print(zipper.get_compressed(postings))

# unzip = GammaCodeDecompressor()
# print(unzip.bit_stream)
# print(unzip.get_positional_posting_list())
