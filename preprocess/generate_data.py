import os
import csv
import argparse
import numpy as np
from itertools import islice
from PIL import Image

folder_names = {'Training': 'FER2013Train',
                'PublicTest': 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}


def str_to_image(image_blob):
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return Image.fromarray(image_data)


def main(base_folder, fer_path, ferplus_path):
    print("Start")

    for key, value in folder_names.items():
        folder_path = os.path.join(base_folder, value)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    ferplus_entries = []
    with open(ferplus_path,'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)

    index = 0
    with open(fer_path,'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            if len(file_name) > 0:
                image = str_to_image(row[1])
                ##라벨링 하는과정 넣어서 path 에 추가,folder_names 하고 file_name사이에
                ##fer2013new.csv에 값이 가장큰것을 찾아내서 감정을 분류한다.
                max_value = ferplus_row[2:]
                max_value_index = max_value.index(max(max_value))
                if max_value_index == 0:
                    row[0] = '6' # neutral
                elif max_value_index == 1:
                    row[0] = '3' # happy
                elif max_value_index == 2:
                    row[0] = '5' # surprise
                elif max_value_index == 3:
                    row[0] = '4' # sad
                elif max_value_index == 4:
                    row[0] = '0' # angry
                elif max_value_index == 5:
                    row[0] = '1' # disgust
                elif max_value_index == 6:
                    row[0] = '2' # fear
                elif max_value_index == 7:
                    row[0] = '7' # contempt
                elif max_value_index == 8:
                    row[0] = '8' # unknown
                elif max_value_index == 9:
                    row[0] = '9' # X
                image_path = os.path.join(base_folder, folder_names[row[2]], row[0], file_name)
                image.save(image_path, compress_level=0)
            index += 1

    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--base_folder",
                        type = str,
                        help = "Base folder containing the training, validation and testing folder.",
                        required = True)
    parser.add_argument("-fer",
                        "--fer_path",
                        type = str,
                        help = "Path to the original fer2013.csv file.",
                        required = True)

    parser.add_argument("-ferplus",
                        "--ferplus_path",
                        type = str,
                        help = "Path to the new fer2013new.csv file.",
                        required = True)

    args = parser.parse_args()
    main(args.base_folder, args.fer_path, args.ferplus_path)


###data 폴더안에, FER2013Test,FER2013Valid,FER2013Train폴더
###python generate_data.py -d ../data -fer ../data./fer2013.csv -ferplus ../data./fer2013new.csv
