import numpy as np
from collections import Counter
import PIL

from Compressors import Huffman as Huf


"""Это изначальный файл, где я подгтавливал функции"""


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


def to_ycbcr(rgb_image: np.array):
    R = rgb_image[:, :, 0].astype(np.float32)
    G = rgb_image[:, :, 1].astype(np.float32)
    B = rgb_image[:, :, 2].astype(np.float32)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    ycbcr_image = np.stack([Y, Cb, Cr], axis=-1).clip(0, 255).astype(np.uint8)
    return ycbcr_image


def from_ycbcr(ycbcr_image: np.array):
    Y = ycbcr_image[:, :, 0].astype(np.float32)
    Cb = ycbcr_image[:, :, 1].astype(np.float32) - 128
    Cr = ycbcr_image[:, :, 2].astype(np.float32) - 128

    R = Y + 1.402 * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.772 * Cb

    rgb_image = np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)
    return rgb_image


def downsampling(channel: np.array):
    h, w = channel.shape
    downsampled = np.zeros((h // 2, w // 2))

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = channel[i:i + 2, j:j + 2]
            downsampled[i // 2, j // 2] = np.mean(block)

    return downsampled.astype(np.uint8)


def idownsampling(channel: np.array):
    h, w = channel.shape
    restored = np.zeros((h * 2, w * 2))

    for i in range(h):
        for j in range(w):
            restored[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = channel[i, j]

    return restored.astype(np.uint8)


def split_blocks(matrix: np.array, n=8):
    h, w, *_ = matrix.shape
    if h % n != 0 or w % n != 0:
        new_h = n * int(np.ceil(h / n))
        new_w = n * int(np.ceil(w / n))

        new_image = np.full((new_h, new_w), 0, dtype=object)
        new_image[:h, :w] = matrix

        matrix = new_image
        h, w = new_h, new_w

    blocks_array = np.array([matrix[i:i + n, j: j + n] for i in range(0, h, n) for j in range(0, w, n)])
    return blocks_array


def join_blocks(blocks: np.array, h, w, n=8):
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


def c_dct(u):
    if u == 0:
        return 2 ** -0.5
    return 1


def dct(matrix: np.array):
    def F(u, v):
        n = matrix.shape[0]
        res = 0.0
        for x in range(n):
            for y in range(n):
                cos1 = np.cos((2 * x + 1) * u * np.pi / (2 * n))
                cos2 = np.cos((2 * y + 1) * v * np.pi / (2 * n))
                res += matrix[x, y] * cos1 * cos2

        res = res * c_dct(u) * c_dct(v) * 2 / n
        return res

    h, w = matrix.shape[:2]
    if h != w:
        raise ValueError(f"Матрица должна быть квадратной: h={h}, w={w}")
    n_ = h
    return np.array([[F(i, j) for j in range(n_)] for i in range(n_)])


def idct(matrix: np.array):
    def f(x, y):
        n = matrix.shape[0]
        res = 0.0
        for u in range(n):
            for v in range(n):
                cos1 = np.cos((2 * x + 1) * u * np.pi / (2 * n))
                cos2 = np.cos((2 * y + 1) * v * np.pi / (2 * n))
                res += c_dct(u) * c_dct(v) * matrix[u, v] * cos1 * cos2

        res = res * 2 / n
        return res

    h, w = matrix.shape[:2]
    if h != w:
        raise ValueError(f"Матрица должна быть квадратной: h={h}, w={w}")
    n_ = h
    return np.array([[f(i, j) for j in range(n_)] for i in range(n_)])


def quant_matrix(qmatrix, scale):
    scale = min(max(scale, 1), 100)
    if scale < 50:
        scale = 5000 / scale
    else:
        scale = 200 - 2 * scale
    scale = np.clip(scale / 100, 0.01, 255)

    new_qmatrix = np.round(qmatrix * scale).astype(np.int32)
    new_qmatrix = np.clip(new_qmatrix, 1, 255)
    return new_qmatrix


def quantize_block(block, qmatrix):
    return np.round(block / qmatrix).astype(np.int32)


def restore_block(block, qmatrix):
    return np.round(block * qmatrix).astype(np.int32)


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


def category(delta):
    abs_val = abs(delta)
    if abs_val == 0:
        return 0
    return int(np.floor(np.log2(abs_val))) + 1


def convert(x, cat):
    if x < 0:
        x = abs(x) ^ (1 << cat - 1)
    return bin(x)[2:].zfill(cat)

def iconvert(bits, cat):
    value = int(bits, 2)
    if bits[0] == '0':
        value = -((1 << cat) - 1 - value)
    return value

def encode_dc(arr: np.array):
    n = len(arr)
    arr = tuple(map(int, arr))

    arr_dc = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
    categories = tuple(map(category, arr_dc))
    freq_dict = dict(Counter(categories))

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

    return bytes(encoded), freq_dict, padding


def encode_ac(arr: np.array):
    n = len(arr)
    arr = tuple(map(int, arr))
    arr_ac = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
    ac, rle_ac = [], []

    start = 0
    for i in range(n):
        if arr_ac[i] != 0:
            ac.append(arr_ac[i])
            rle_ac.append((i - start, category(ac[i])))
            start = i + 1

    freq_dict = dict(Counter(rle_ac))
    root = Huf.build_tree(freq_dict)
    codes = Huf.build_code(root)

    huf_str = ""
    encoded = bytearray()
    for i in range(len(ac)):
        cat = rle_ac[i][1]
        huf_str += codes[rle_ac[i]] + convert(ac[i], cat)

        while len(huf_str) >= 8:
            encoded.append(int(huf_str[:8], 2))
            huf_str = huf_str[8:]

    padding = 0
    if len(huf_str) != 0:
        padding = 8 - len(huf_str)
        encoded.append(int(huf_str.ljust(8, '0'), 2))

    return bytes(encoded), freq_dict, padding


def decode_dc(bits_buffer, cat, decoded):
    dc = iconvert(bits_buffer[:cat], cat)
    decoded.append(dc + decoded[-1] if decoded else dc)
    return cat

def decoded_ac(bits_buffer, value, decoded):
    run_len, cat = value
    ac = iconvert(bits_buffer[:cat], cat)

    decoded.extend([0] * run_len)
    decoded.append(ac)
    return cat


def decode(data: bytes, root: Huf.Node, padding: int, decode_func, node=None):
    decoded = []

    bits_buffer = ""
    cur_node = node if node else root

    for b in data:
        bits_buffer += bin(b)[2:].rjust(8, '0')

    if 0 < padding < 8:
        bits_buffer = bits_buffer[:-padding]
    while len(bits_buffer) > 0:
        if cur_node.value is not None:
            step = decode_func(bits_buffer, cur_node.value, decoded)

            bits_buffer = bits_buffer[step:]
            cur_node = root
        if not bits_buffer:
            break

        bit = bits_buffer[0]
        bits_buffer = bits_buffer[1:]
        if bit == '0':
            cur_node = cur_node.left
        else:
            cur_node = cur_node.right

    # if cur_node and cur_node.value is not None:
    #     cat = cur_node.value
    #     dc = iconvert(bits_buffer[:cat], cat)
    #     decoded.append(dc)

    return decoded, cur_node


rgb_pixels = np.array([
    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 1)],
    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 1)]
])

# pixels = to_ycbcr(rgb_pixels)

grayscale_block = np.array([
    [50, 60, 55, 58, 52, 53, 57, 59],
    [62, 65, 59, 61, 54, 56, 58, 60],
    [48, 52, 50, 53, 49, 51, 55, 57],
    [55, 58, 56, 59, 53, 54, 56, 58],
    [51, 53, 52, 54, 50, 52, 54, 56],
    [57, 59, 58, 60, 55, 56, 58, 60],
    [49, 51, 50, 52, 48, 50, 52, 54],
    [53, 55, 54, 56, 52, 53, 55, 57]
], dtype=np.uint8)

