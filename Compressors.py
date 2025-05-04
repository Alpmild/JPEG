import heapq

class Huffman:
    folder = "Files\\Results\\Huffman"

    class Node:
        def __init__(self, value=None, freq=0, left=None, right=None):
            self.value = value
            self.freq = freq
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self, reps_size=4):
        self.count_size = 2
        self.reps_size = reps_size
        self.step = self.reps_size + 1

    @classmethod
    def build_tree(cls, freq_dict: dict):
        nodes = [cls.Node(value=i, freq=freq_dict[i]) for i in freq_dict.keys() if freq_dict[i]]
        nodes.sort(key=lambda x: x.value)
        nodes.sort(key=lambda x: x.freq)
        heapq.heapify(nodes)

        while len(nodes) != 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            heapq.heappush(nodes, cls.Node(value=None, freq=left.freq + right.freq, left=left, right=right))
        return heapq.heappop(nodes)

    @classmethod
    def build_code(cls, node, prefix="", code_dict=None):
        if code_dict is None:
            code_dict = dict()
        if node:
            if node.value is not None:
                code_dict[node.value] = prefix
            else:
                cls.build_code(node.left, prefix + '0', code_dict)
                cls.build_code(node.right, prefix + '1', code_dict)
        return code_dict

    # @staticmethod
    # def encode(data: bytes, codes: dict, prefix: str):
    #     encoded = bytearray()
    #     encoded_bits = prefix
    #     for byte in data:
    #         encoded_bits += codes[bytes([byte])]
    #
    #     while len(encoded_bits) >= 8:
    #         byte = int(encoded_bits[:8], 2)
    #         encoded.extend(bytes([byte]))
    #         encoded_bits = encoded_bits[8:]
    #
    #     return bytes(encoded), encoded_bits
    #
    # @staticmethod
    # def decode(data: bytes, root, sign_bits: int, node):
    #     decoded = bytearray()
    #
    #     bits_buffer = ""
    #     cur_node = node
    #
    #     for b in data:
    #         bits_buffer += bin(b)[2:].rjust(8, '0')
    #
    #     if 0 < sign_bits < 8:
    #         bits_buffer = bits_buffer[:-8 + sign_bits]
    #     while len(bits_buffer) > 0:
    #         if cur_node.value is not None:
    #             decoded.extend(cur_node.value)
    #             cur_node = root
    #         if not bits_buffer:
    #             break
    #
    #         bit = bits_buffer[0]
    #         bits_buffer = bits_buffer[1:]
    #         if bit == '0':
    #             cur_node = cur_node.left
    #         else:
    #             cur_node = cur_node.right
    #     if cur_node and cur_node.value is not None:
    #         decoded.extend(cur_node.value)
    #         cur_node = root
    #     return bytes(decoded), cur_node

    # def encode_file(self, input_file, output_file):
    #     freq_dict = dict()
    #
    #     with open(input_file, "rb") as ifile:
    #         while True:
    #             block = ifile.read(self.block)
    #             if not block:
    #                 break
    #             for i in block:
    #                 i = bytes([i])
    #                 if i not in freq_dict.keys():
    #                     freq_dict[i] = 0
    #                 freq_dict[i] += 1
    #
    #     root = self.build_tree(freq_dict)
    #     codes = self.build_code(root)
    #
    #     with open(input_file, "rb") as ifile, open(output_file, "wb") as ofile:
    #         # Запись количества байт под размер блока
    #         ofile.write(bytes([BLOCK_BYTES]))
    #         # Запись размера блока
    #         ofile.write(self.block.to_bytes(BLOCK_BYTES, byteorder="big"))
    #
    #         # Запись количества байт под количество ключей в словаре частот
    #         ofile.write(self.count_size.to_bytes(1, byteorder="big"))
    #         # Запись количества повторений
    #         ofile.write(len(freq_dict).to_bytes(self.count_size, byteorder="big"))
    #         # Запись количества байт под количество повторений
    #         ofile.write(self.reps_size.to_bytes(1, byteorder="big"))
    #
    #         # Запись словаря
    #         for byte, freq in freq_dict.items():
    #             ofile.write(byte)
    #             ofile.write(freq.to_bytes(self.reps_size, byteorder="big"))
    #
    #         prefix = ""
    #         while True:
    #             block = ifile.read(self.block)
    #             if not block:
    #                 break
    #             encoded, prefix = self.encode(block, codes, prefix)
    #             ofile.write(encoded)
    #
    #         if prefix:
    #             sign_bits = len(prefix)
    #             byte = int(prefix.ljust(8, "0"), 2)
    #             ofile.write(bytes([byte, sign_bits]))
    #         else:
    #             ofile.write(bytes([8]))
    #
    # def decode_file(self, input_file, output_file):
    #     freq_dict = dict()
    #     ifile_size = os.path.getsize(input_file)
    #
    #     with open(input_file, "rb") as ifile:
    #         block_cnt = int.from_bytes(ifile.read(1), byteorder="big")
    #         block_bytes = int.from_bytes(ifile.read(block_cnt), byteorder="big")
    #
    #         count_size = int.from_bytes(ifile.read(1), byteorder="big")
    #         count = int.from_bytes(ifile.read(count_size), byteorder="big")
    #         reps_size = int.from_bytes(ifile.read(1), byteorder="big")
    #
    #         for _ in range(count):
    #             byte = ifile.read(1)
    #             freq = int.from_bytes(ifile.read(reps_size), byteorder="big")
    #             freq_dict[byte] = freq
    #         root = self.build_tree(freq_dict)
    #
    #         pos = ifile.tell()
    #         ifile.seek(-1, 2)
    #         sign_bits = int.from_bytes(ifile.read(1), byteorder="big")
    #         ifile.seek(pos, 0)
    #
    #         node = root
    #         with open(output_file, "wb") as ofile:
    #             while True:
    #                 block = ifile.read(block_bytes)
    #                 if not block:
    #                     break
    #                 s_bits = 0
    #                 if ifile.tell() == ifile_size:
    #                     s_bits = sign_bits
    #                     block = block[:-1]
    #                 if block:
    #                     decoded, node = self.decode(block, root, s_bits, node)
    #                     ofile.write(decoded)
