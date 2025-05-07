import os
from PIL import Image
import numpy as np
from matplotlib import pyplot

files = {
    "Lenna.raw": range(0, 101, 5),
    "City.raw": range(0, 101, 5),
    "Cyberpunk.raw": range(0, 101, 5),
    "gray_Cyberpunk.raw": range(0, 101, 5),
    "bw_dithered_Cyberpunk.raw": range(0, 101, 5),
    "bw_no_dither_Cyberpunk.raw": range(0, 101, 5)
}


def conver_wb(file):
    mode = "1"

    name, exp = os.path.splitext(file)
    folder = f"Images\\Restored\\{name}"
    quality = range(5, 101, 5)

    if not os.path.exists(name):
        os.mkdir(name)
    for q in quality:
        path = f"{folder}\\{name}_{q}{exp}"
        im = Image.open(path).convert(mode)
        im.save(f"{name}\\{name}_{q}{exp}")

    image = Image.open("C:\\Users\\LaRDe\\Downloads\\a.jpg")

    # Сохраняем в PNG
    image.save("Images\\Originals\\City.png")

    print("Конвертация завершена!")

    filename = "Images\\Originals\\Cyberpunk.png"
    output_folder = "Images\\Originals\\"

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(filename)

        # Grayscale
        gray = img.convert("L")
        gray.save(f"{output_folder}gray_Cyberpunk.png")

        # Ч/Б без дизеринга
        bw = img.convert("1")
        bw.save(f"{output_folder}bw_no_dither_Cyberpunk.png")

        # Ч/Б с дизерингом
        dithered = img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)
        dithered.save(f"{output_folder}bw_dithered_Cyberpunk.png")


def grath(file, quality):
    o_y, t_y, r_y = [], [], []

    name, exp = os.path.splitext(file)

    orig_fold = "Images\\RawOriginals\\"
    temp_fold = f"Images\\Temp\\{name}\\"
    res_fold = f"Images\\Restored\\{name}\\"

    o_size = os.path.getsize(orig_fold + file) / 1024
    for q in quality:
        # o_y.append(o_size)

        t_size = os.path.getsize(temp_fold + f"{name}_{q}") / 1024
        t_y.append(t_size)

        # try:
        #     r_size = round(os.path.getsize(res_fold + f"{name}_{q}{exp}") / 1024, 1)
        #     r_y.append(r_size)
        # except FileNotFoundError:
        #     continue

        print(q, t_size)

    pyplot.figure(figsize=(10, 6))

    # pyplot.plot(quality, o_y, '-o', label="Orig.", color="red")
    pyplot.plot(quality, t_y, '-o', label="Temp.", color="blue")
    # pyplot.plot(quality, r_y, '-o', label="Res.", color="red")

    pyplot.title(file)
    pyplot.xlabel("Качество, %")
    pyplot.ylabel("Сжатие")
    pyplot.grid()
    pyplot.legend()

    pyplot.savefig(f"Графики\\{name}.png")
    pyplot.close()


def save_raw(img_path: str) -> None:
    img = Image.open(f"Images\\Originals\\{img_path}").convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    root, _ = os.path.splitext(img_path)
    raw_path = f"Fold\\{root}.raw"

    with open(raw_path, 'wb') as f:
        f.write(img_array.tobytes())


for f, q in files.items():
    grath(f, q)