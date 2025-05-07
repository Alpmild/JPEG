from PIL import Image
import struct
import numpy as np
from collections import Counter

from Compressors import Huffman as Huf

CAT_FORM = '>BI'
RLE_CAT_FORM = '>IBI'
SEQ_LEN_FORM = '>I'
INFO_FORM = '>HHBBB'
N = 8
f = 0

modes = ['RGB', 'L', '1', 'RGBA']

Y_QMATRIX = np.array([
    [16,  11,  10,  16,  24,  40,  51,  61],
    [12,  12,  14,  19,  26,  58,  60,  55],
    [14,  13,  16,  24,  40,  57,  69,  56],
    [14,  17,  22,  29,  51,  87,  80,  62],
    [18,  22,  37,  56,  68, 109, 103,  77],
    [24,  35,  55,  64,  81, 104, 113,  92],
    [49,  64,  78,  87, 103, 121, 120, 101],
    [72,  92,  95,  98, 112, 100, 103,  99]
])
CbCr_QMATRIX = np.array([
    [17,  18,  24,  47,  99,  99,  99,  99],
    [18,  21,  26,  66,  99,  99,  99,  99],
    [24,  26,  56,  99,  99,  99,  99,  99],
    [47,  66,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99]
])


def c_dct(u):
    if isinstance(u, int):
        return 2 ** -0.5 if u == 0 else 1.0
    u = np.asarray(u)
    return np.where(u == 0, 2 ** -0.5, 1.0)


def category(delta):
    abs_val = abs(delta)
    if abs_val == 0:
        return 0
    return int(np.floor(np.log2(abs_val))) + 1


def convert(x, cat):
    if cat == 0:
        return ""
    if x < 0:
        x = ((1 << cat) - 1) ^ (abs(x))
    return bin(x)[2:].zfill(cat)


def iconvert(bits, cat):
    if cat == 0:
        return 0
    value = int(bits, 2)
    if bits[0] == '0':
        value = -((1 << cat) - 1 - value)
    return value


def quant_matrix(qmatrix, quality):
    if quality <= 0:
        quality = 0.01
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    scaled_matrix = ((qmatrix * scale + 50) / 100).astype(np.uint8)
    scaled_matrix[scaled_matrix == 0] = 1
    return scaled_matrix


def decode(data: bytes, root: Huf.Node, padding: int, mode: str):
    assert mode == 'AC' or mode == 'DC'

    bits_buffer = ''.join(f'{byte:08b}' for byte in data)
    if padding > 0:
        bits_buffer = bits_buffer[:-padding]

    decoded = []
    cur_node = root
    i = 0

    while i < len(bits_buffer):
        bit = bits_buffer[i]
        i += 1
        cur_node = cur_node.left if bit == '0' else cur_node.right

        if cur_node.value is not None:
            if mode == 'DC':
                cat = cur_node.value
                if cat != 0:
                    dc = iconvert(bits_buffer[i:i + cat], cat)
                    decoded.append(dc)
                else:
                    decoded.append(0)
                i += cat
            else:
                run_len, cat = cur_node.value
                decoded.extend([0] * run_len)
                if cat != 0:
                    ac = iconvert(bits_buffer[i:i + cat], cat)
                    decoded.append(ac)
                else:
                    break
                i += cat
            cur_node = root

    for i in range(1, len(decoded)):
        decoded[i] += decoded[i - 1]

    return np.array(decoded)


class JPEGencoder:
    def __init__(self, path, scale):
        image = Image.open(path)
        self.mode = modes.index(image.mode)
        if image.mode in ('L', '1'):
            image = image.convert('RGB')
        self.image = np.array(image)

        self.scale = scale
        self.y_qmatrix = quant_matrix(Y_QMATRIX, scale)
        self.cbcr_qmatrix = quant_matrix(CbCr_QMATRIX, scale)

    @staticmethod
    def to_ycbcr(rgb_image: np.array):
        R = rgb_image[:, :, 0].astype(np.float32)
        G = rgb_image[:, :, 1].astype(np.float32)
        B = rgb_image[:, :, 2].astype(np.float32)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

        ycbcr_image = np.stack([Y, Cb, Cr], axis=-1).clip(0, 255).astype(np.uint8)
        return ycbcr_image

    @staticmethod
    def downsampling(channel: np.array):
        h, w = channel.shape
        downsampled = np.zeros((h // 2, w // 2))

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                block = channel[i:i + 2, j:j + 2]
                downsampled[i // 2, j // 2] = np.mean(block)

        return downsampled.astype(np.uint8)

    @staticmethod
    def split_blocks(image_arr: np.array, n=N):
        h, w, *_ = image_arr.shape
        if h % n != 0 or w % n != 0:
            new_h = n * (h // n + bool(h % n))
            new_w = n * (w // n + bool(w % n))

            new_image = np.full((new_h, new_w), 0, dtype=object)
            new_image[:h, :w] = image_arr

            image_arr = new_image
            h, w = new_h, new_w

        blocks_array = np.array([image_arr[i:i + n, j: j + n] for i in range(0, h, n) for j in range(0, w, n)])
        return blocks_array

    @staticmethod
    def dct(block: np.ndarray):
        def F(u, v):
            n = block.shape[0]
            x = np.arange(n)
            y = np.arange(n)

            cos1 = np.cos((2 * x[:, None] + 1) * u * np.pi / (2 * n))
            cos2 = np.cos((2 * y[None, :] + 1) * v * np.pi / (2 * n))

            cos_matrix = cos1 @ cos2
            res = np.sum(block * cos_matrix)

            res *= c_dct(u) * c_dct(v) * 2 / n
            return res

        h, w = block.shape[:2]
        if h != w:
            raise ValueError(f"Матрица должна быть квадратной: h={h}, w={w}")
        n_ = h
        return np.array([[F(i, j) for j in range(n_)] for i in range(n_)])

    @staticmethod
    def quantize_block(block, qmatrix):
        return np.round(block / qmatrix).astype(np.int32)

    @staticmethod
    def zigzag(matrix):
        n = len(matrix)
        result = []
        i, j = 0, 0

        for _ in range(n * n):
            result.append(matrix[i][j])

            if (i + j) % 2 == 0:  # Движение вверх-вправо
                if j == n - 1:
                    i += 1
                elif i == 0:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Движение вниз-влево
                if i == n - 1:
                    j += 1
                elif j == 0:
                    i += 1
                else:
                    i += 1
                    j -= 1

        return np.array(result)

    @staticmethod
    def encode_dc(arr: np.array, file):
        n = len(arr)
        arr = tuple(map(int, arr))

        arr_dc = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
        categories = tuple(map(category, arr_dc))
        freq_dict = dict(Counter(categories))

        # Запись количества категорий
        file.write(struct.pack('>H', len(freq_dict)))
        # Запись словаря частотностей
        for i in sorted(freq_dict.keys()):
            file.write(struct.pack(CAT_FORM, i, freq_dict[i]))

        root = Huf.build_tree(freq_dict)
        codes = Huf.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(n):
            dc = arr_dc[i]
            cat = category(dc)

            huf_str += codes[cat] + convert(dc, cat)
            while len(huf_str) >= 8:
                encoded.append(int(huf_str[:8], 2))
                huf_str = huf_str[8:]

        padding = 0
        if len(huf_str) != 0:
            padding = 8 - len(huf_str)
            encoded.append(int(huf_str.ljust(8, '0'), 2))

        # Запись длины байтового потока
        file.write(struct.pack(SEQ_LEN_FORM, len(encoded)))
        # Байтовый поток
        file.write(bytes(encoded))
        # Число нулей, добавленных в конец
        file.write(struct.pack('>B', padding))

    @staticmethod
    def encode_ac(arr: np.array, file):
        n = len(arr)
        arr = tuple(map(int, arr))
        arr_ac = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
        ac, rle_ac = [], []

        zeros_cnt = 0
        for i in range(n):
            if arr_ac[i] != 0:
                ac.append(arr_ac[i])
                rle_ac.append((zeros_cnt, category(arr_ac[i])))
                zeros_cnt = 0
            else:
                zeros_cnt += 1
        if zeros_cnt != 0:
            ac.append(0)
            rle_ac.append((zeros_cnt, 0))

        freq_dict = dict(Counter(rle_ac))
        # Запись количества пар
        file.write(struct.pack('>H', len(freq_dict)))
        # Запись словаря частотностей
        for couple, value in sorted(freq_dict.items()):
            file.write(struct.pack(RLE_CAT_FORM, *couple, value))

        root = Huf.build_tree(freq_dict)
        codes = Huf.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(len(rle_ac)):
            cat = rle_ac[i][1]
            huf_str += codes[rle_ac[i]] + convert(ac[i], cat)

            while len(huf_str) >= 8:
                encoded.append(int(huf_str[:8], 2))
                huf_str = huf_str[8:]

        padding = 0
        if len(huf_str) != 0:
            padding = 8 - len(huf_str)
            encoded.append(int(huf_str.ljust(8, '0'), 2))

        # Запись длины байтового потока
        file.write(struct.pack(SEQ_LEN_FORM, len(encoded)))
        # Байтовый поток
        file.write(bytes(encoded))
        # Число нулей, добавленных в конец
        file.write(struct.pack('>B', padding))

    def process(self, path, block_size=N):
        h, w = self.image.shape[:2]
        output_file = open(path, "wb")
        output_file.write(struct.pack(INFO_FORM, h, w, self.mode, block_size, self.scale))

        image_ycbcr = self.to_ycbcr(self.image)
        for i in range(3):
            channel = image_ycbcr[:, :, i]
            if i != 0:
                channel = self.downsampling(channel)

            blocks = self.split_blocks(channel, block_size)
            blocks = np.array(tuple(map(self.dct, blocks)))

            qmatrix = self.y_qmatrix if i == 0 else self.cbcr_qmatrix
            blocks = np.array(tuple(map(lambda x: self.quantize_block(x, qmatrix), blocks)))
            blocks = np.array(tuple(map(lambda x: self.zigzag(x), blocks)))

            dc = blocks[:, 0]
            print("DC", len(dc))
            self.encode_dc(dc, output_file)

            ac = np.hstack(blocks[:, 1:])
            print("AC", len(ac))
            self.encode_ac(ac, output_file)

        output_file.close()


class JPEGdecoder:
    def __init__(self, path):
        self.image = open(path, "rb")

    @staticmethod
    def from_ycbcr(ycbcr_image: np.array):
        Y = ycbcr_image[:, :, 0].astype(np.float32)
        Cb = ycbcr_image[:, :, 1].astype(np.float32) - 128
        Cr = ycbcr_image[:, :, 2].astype(np.float32) - 128

        R = Y + 1.402 * Cr
        G = Y - 0.34414 * Cb - 0.71414 * Cr
        B = Y + 1.772 * Cb

        rgb_image = np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)
        return rgb_image

    @staticmethod
    def idownsampling(channel: np.array):
        h, w = channel.shape
        restored = np.zeros((h * 2, w * 2))

        for i in range(h):
            for j in range(w):
                restored[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = channel[i, j]

        return restored.astype(np.uint8)

    @staticmethod
    def join_blocks(blocks: np.array, h, w, n=N):
        new_h, new_w = h, w
        if h % n != 0 or w % n != 0:
            new_h = n * int(np.ceil(h / n))
            new_w = n * int(np.ceil(w / n))

        image = np.full((new_h, new_w), 0, dtype=object)
        u, v = new_h // n, new_w // n
        for i in range(u):
            for j in range(v):
                for k in range(n):
                    image[i * n + k][j * n: j * n + n] = blocks[i * v + j][k]
        return image[:h, :w]

    @staticmethod
    def idct(block: np.ndarray):
        def F(x, y):
            n = block.shape[0]
            u = np.arange(n)
            v = np.arange(n)

            cos1 = np.cos((2 * x + 1) * u[:, None] * np.pi / (2 * n))
            cos2 = np.cos((2 * y + 1) * v[:, None] * np.pi / (2 * n))

            cos_matrix = cos1 @ cos2.T

            c_u = c_dct(u).reshape(-1, 1)
            c_v = c_dct(v).reshape(1, -1)
            scale = c_u * c_v

            res = np.sum(scale * block * cos_matrix)
            res *= 2 / n
            return res

        h, w = block.shape[:2]
        if h != w:
            raise ValueError(f"Матрица должна быть квадратной: h={h}, w={w}")
        n_ = h
        return np.array([[F(i, j) for j in range(n_)] for i in range(n_)])

    @staticmethod
    def restore_block(block, qmatrix):
        return np.round(block * qmatrix).astype(np.int32)

    @staticmethod
    def inverse_zigzag(arr, n=8):
        matrix = [[0] * n for _ in range(n)]
        i, j = 0, 0

        for idx in range(n * n):
            matrix[i][j] = arr[idx]

            if (i + j) % 2 == 0:  # Движение вверх-вправо
                if j == n - 1:
                    i += 1
                elif i == 0:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Движение вниз-влево
                if i == n - 1:
                    j += 1
                elif j == 0:
                    i += 1
                else:
                    i += 1
                    j -= 1

        return np.array(matrix)

    @staticmethod
    def decode(file, mode):
        assert mode == 'AC' or mode == 'DC'

        freg_dict_len = struct.unpack('>H', file.read(2))[0]
        freq_dict = dict()
        for i in range(freg_dict_len):
            form = CAT_FORM if mode == 'DC' else RLE_CAT_FORM
            s = struct.calcsize(form)
            couple = struct.unpack(form, file.read(s))

            key, value = couple if mode == 'DC' else (couple[:2], couple[2])
            freq_dict[key] = value

        len_data = struct.unpack(SEQ_LEN_FORM, file.read(struct.calcsize(SEQ_LEN_FORM)))[0]
        data = file.read(len_data)
        padding = struct.unpack('>B', file.read(1))[0]

        root = Huf.build_tree(freq_dict)
        return decode(data, root=root, padding=padding, mode=mode)

    def process(self, path):
        size = struct.calcsize(INFO_FORM)
        h, w, mode, block_size, scale = struct.unpack(INFO_FORM, self.image.read(size))
        mode = modes[mode]

        k = block_size ** 2 - 1
        yb_cnt = int(np.ceil(h / block_size) * np.ceil(h / block_size))
        cbcrb_cnt = int(np.ceil(h // 2 / block_size) * np.ceil(h // 2 / block_size))

        y_qmatrix = quant_matrix(Y_QMATRIX, scale)
        cbcr_qmatrix = quant_matrix(CbCr_QMATRIX, scale)

        channels = tuple()
        for i in range(3):
            if i == 0:
                b = yb_cnt
                qmatrix = y_qmatrix
            else:
                b = cbcrb_cnt
                qmatrix = cbcr_qmatrix
            dc = np.array(self.decode(self.image, 'DC'))
            print("DC", len(dc))
            dc.resize(b, 1)

            ac = np.array(self.decode(self.image, 'AC'))
            print("AC", len(ac))
            ac.resize(b, k)

            blocks = tuple(map(lambda x: np.array(self.inverse_zigzag(np.concatenate((dc[x], ac[x])), block_size)), range(b)))
            blocks = np.array(tuple(map(lambda x: self.restore_block(x, qmatrix), blocks)))

            blocks = np.array(tuple(map(self.idct, blocks)))
            if i != 0:
                channel = self.join_blocks(blocks, h // 2, w // 2, block_size)
                channel = self.idownsampling(channel)
            else:
                channel = self.join_blocks(blocks, h, w, block_size)

            channels += (channel,)

        new_image = np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)
        new_image = Image.fromarray(self.from_ycbcr(new_image))
        if mode != 'RGB':
            new_image = new_image.convert(mode)

        new_image.save(path)
