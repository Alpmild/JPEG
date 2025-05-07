import os
from main import *

input_folder = "Images\\Originals\\"
temp_folder = "Images\\Temp\\"
output_folder = "Images\\Restored\\"

files = {
    # "Lenna.png": range(0, 1),
    "City.png": range(10, 101, 10),
    # "Cyberpunk.png": range(0,  1),
    # "gray_Cyberpunk.png": range(0, 1),
    # "bw_dithered_Cyberpunk.png": range(0, 1),
    # "bw_no_dither_Cyberpunk.png": range(0, 1)
}

for file in files:
    for s in files[file]:
        print(file, s)
        file_name, exp = os.path.splitext(file)

        tfold = temp_folder + file_name
        if not os.path.exists(tfold):
            os.mkdir(tfold)
        ofold = output_folder + file_name
        if not os.path.exists(ofold):
            os.mkdir(ofold)

        temp_file = f"{tfold}\\{file_name}_{s}"
        encoder = JPEGencoder(input_folder + file, s)
        encoder.process(temp_file)

        output_file = f"{ofold}\\{file_name}_{s}.png"
        decoder = JPEGdecoder(temp_file)
        decoder.process(output_file)
