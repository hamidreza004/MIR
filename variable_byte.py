class VariableByteCompressor:

    def get_compressed(self, positional_posting_list):
        compressed = self.compress(positional_posting_list)
        padded_compressed = self.set_padding(compressed)
        return self.to_byte(padded_compressed)

    def compress(self, positional_posting_list):
        compressed = ""
        last_document_number = 0
        for i in range(0, len(positional_posting_list)):
            if i % 2 == 0:
                # compress document number
                document_number = positional_posting_list[i] + 1
                compressed += self.get_gamma_code(document_number - last_document_number)
                last_document_number = document_number
            else:
                # compress document number
                positions = positional_posting_list[i] + 1
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

    def to_byte(self, bit_stream):
        byte_stream = ""
        i = 0
        while i < len(bit_stream):
            byte_stream += chr(int(bit_stream[i: i + 8], 2))
            i += 8
        return byte_stream

    def get_gamma_code(self, decimal_number):
        offset = self.get_binary_code(decimal_number)[1:]
        length = self.get_unary_code(len(offset))
        return length + offset

    def get_unary_code(self, decimal_number):
        unary_code = ""
        for i in range(0, decimal_number):
            unary_code += '1'
        unary_code += '0'
        return unary_code

    def get_binary_code(self, deciaml_number):
        return bin(deciaml_number)[2:]

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

    def __init__(self):
        self.bit_stream = ""
        self.iterator = 3

    def get_decompressed(self, byte_stream):
        self.bit_stream = self.to_bit(byte_stream)
        return self.get_positional_posting_list()

    def get_positional_posting_list(self):
        posting_list = []
        document_number = -1
        document_diff = self.get_next_number()
        while document_diff is not None:
            document_number += document_diff
            posting_list.append(document_number)
            positions_len = self.get_next_number()
            positions = []
            position = -1
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
            if self.iterator == len(self.bit_stream):
                return None
        self.iterator += 1
        offset = self.bit_stream[self.iterator: self.iterator + length]
        self.iterator += length
        return int('1' + offset, 2)

    def to_bit(self, byte_stream):
        bit_stream = ""
        for byte in byte_stream:
            bit_stream += format(ord(byte), '08b')
        return bit_stream
