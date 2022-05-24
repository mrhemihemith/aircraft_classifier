import os
import json
import librosa
from PIL import Image
import cv2
import numpy as np
#from conf import DATA_DIR
DATA_DIR = "Data"

DATA_SET = os.path.join(DATA_DIR, "Chinook")
os.makedirs(DATA_SET, exist_ok=True)

DATA_SET = os.path.join(DATA_DIR, "Helicopters")
os.makedirs(DATA_SET, exist_ok=True)

DATA_SET = os.path.join(DATA_DIR, "Planes")
os.makedirs(DATA_SET, exist_ok=True)

DATA_SET = 'img_data'
JSON_PATH = 'image_test_tst.json'
NEW_DATA_SET = 'DATA'


def PREPARE_DATASET(data_set, json_path, new_data_set):
    data = {
        'mappings': [],
        'labels': [],
        'files': [],
        'PIXs': []

    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_set)):
        if dirpath is not data_set:
            category = dirpath.split('/')[0]
            # data['mappings'].append(category)
            print(f'Processing : {category}')

            print(f'dirpath : {dirpath}')
            print(f'dirnames : {dirnames}')
            print(f'filenames : {filenames}')

            if dirpath == 'Data/Chinook':
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    chinook_img = Image.open(file_path)
                    new_chinook_img = chinook_img.resize((100, 100))
                    new_chinook_img.save(f'Data/Chinook/{f}')
                    print(f'old bike img size : {chinook_img.size}')
                    print(f'new bike img size : {new_chinook_img.size}')

            elif dirpath == 'Data/Helicopters':

                for f in filenames:
                    file_path = os.path.join(dirpath, f)

                    helicopter_img = Image.open(file_path)

                    new_helicopter_img = helicopter_img.resize((100, 100))

                    new_helicopter_img.save(f'Data/Helicopters/{f}')

                    print(f'old bike img size : {helicopter_img.size}')

                    print(f'new bike img size : {new_helicopter_img.size}')


            elif dirpath == 'Data\Planes':
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    planes_img = Image.open(file_path)
                    new_planes_img = planes_img.resize((100, 100))
                    new_planes_img.save(f'Data/Planes/{f}')
                    print(f'old bike img size : {planes_img.size}')
                    print(f'new bike img size : {new_planes_img.size}')
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(new_data_set)):

        if dirpath is not data_set:
            category = dirpath.split('/')[0]
            data['mappings'].append(category)
            print(f'Processing : {category}')

            print(f'dirpath : {dirpath}')
            print(f'dirnames : {dirnames}')
            print(f'filenames : {filenames}')

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                print(f'file : {file_path}')

                # loading pixels
                img_PIXs = cv2.imread(file_path)

                data['labels'].append(i - 1)
                data['PIXs'].append(img_PIXs.tolist())
                data['files'].append(file_path)
                print(f'File_path: {i - 1}')

            with open(JSON_PATH, 'w') as fp:
                json.dump(data, fp, indent=4)


if __name__ == '__main__':
    PREPARE_DATASET(DATA_SET, JSON_PATH, NEW_DATA_SET)