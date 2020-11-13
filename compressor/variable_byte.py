class VariableByteCompressor:

    def get_compressed(self, posting_list, is_positional=True):
        compressed = self.compress(posting_list, is_positional)
        return self.to_byte(compressed)

    def compress(self, posting_list, is_positional):
        compressed = ""
        last_document_number = -1
        if is_positional:
            for doc in posting_list:
                # compress document number
                document_number = doc[0]
                compressed += self.get_variable_byte(document_number - last_document_number)
                last_document_number = document_number
                # compress position number
                positions = doc[1]
                compressed += self.get_variable_byte(len(positions))
                last_position = -1
                for position in positions:
                    compressed += self.get_variable_byte(position - last_position)
                    last_position = position
        else:
            for document_number in range(posting_list):
                # compress document number
                compressed += self.get_variable_byte(document_number - last_document_number)
                last_document_number = document_number
        return compressed

    def get_variable_byte(self, decimal_number):
        padded_binary = self.get_padded_binary(decimal_number)
        return self.decide_msb(padded_binary)

    def get_padded_binary(self, decimal_number):
        binary = bin(decimal_number)[2:]
        padding_number = (7 - (len(bin(decimal_number)[2:]) % 7)) % 7
        padding = ""
        for i in range(0, padding_number):
            padding += '0'
        return padding + binary

    def decide_msb(self, binary):
        i = 0
        byte_stream = ""
        while i < len(binary):
            if i + 7 < len(binary):
                byte_stream += '0' + binary[i: i + 7]
            else:
                byte_stream += '1' + binary[i: i + 7]
            i += 7
        return byte_stream

    def to_byte(self, bit_stream):
        byte_stream = ""
        i = 0
        while i < len(bit_stream):
            byte_stream += chr(int(bit_stream[i: i + 8], 2))
            i += 8
        return byte_stream


class VariableByteDecompressor:

    def __init__(self):
        self.bit_stream = ""
        self.iterator = 0

    def get_decompressed(self, byte_stream, is_positional=True):
        self.bit_stream = ""
        self.iterator = 0
        self.bit_stream = self.to_bit(byte_stream)
        return self.get_posting_list(is_positional)

    def get_posting_list(self, is_positional):
        posting_list = []
        document_number = -1
        document_diff = self.get_next_number()

        if not is_positional:
            while document_diff is not None:
                document_number += document_diff
                posting_list.append(document_number)
                document_diff = self.get_next_number()
        else:
            while document_diff is not None:
                document_number += document_diff
                positions_len = self.get_next_number()
                positions = []
                position = -1
                for i in range(0, positions_len):
                    position += self.get_next_number()
                    positions.append(position)
                posting_list.append([document_number, positions])
                document_diff = self.get_next_number()
        return posting_list

    def get_next_number(self):
        binary = ""
        if self.iterator == len(self.bit_stream):
            return None
        while self.bit_stream[self.iterator] == '0':
            binary += self.bit_stream[self.iterator + 1:self.iterator + 8]
            self.iterator += 8
        binary += self.bit_stream[self.iterator + 1:self.iterator + 8]
        self.iterator += 8
        return int(binary, 2)

    def to_bit(self, byte_stream):
        bit_stream = ""
        for byte in byte_stream:
            bit_stream += format(ord(byte), '08b')
        return bit_stream

# posting = [[0, [0, 1, 2]], [10, [0, 4, 5, 6, 134]], [15, [0, 1]]]
# zipper = VariableByteCompressor()
#
# unzipper = VariableByteDecompressor()
# print(unzipper.get_decompressed(zipper.get_compressed(posting)))
