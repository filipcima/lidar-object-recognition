import argparse
from os import listdir, rename, path, getcwd
from random import random

DATA_PATH = '../dataset/lidar_2d'
ALL_TXT = '../dataset/image_set/all.txt'
TRAIN_TXT = '../dataset/image_set/train.txt'
TEST_TXT = '../dataset/image_set/test.txt'


def save_items(items, file_path):
    with open(file_path, 'w+') as file:
        file.writelines(['{}\n'.format(item) for item in items])
    print('{}: done'.format(file_path))


def move(item_id, mode):
    origin_path = path.join('..', 'dataset', 'lidar_2d', '{}.npy'.format(item_id))
    destination_path = path.join('..', 'dataset', 'lidar_2d', mode, '{}.npy'.format(item_id))
    rename(origin_path, destination_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ratio', default=getcwd())
    args = parser.parse_args()
    ratio = float(args.ratio)

    all_items, train, test = [], [], []
    for file in listdir(DATA_PATH):
        if file.endswith('.npy'):
            file = file.split('.')[0]
            all_items.append(file)
            if random() <= ratio:
                train.append(file)
                move(file, 'train')
            else:
                test.append(file)
                move(file, 'test')

    save_items(all_items, ALL_TXT)
    save_items(train, TRAIN_TXT)
    save_items(test, TEST_TXT)


if __name__ == '__main__':
    main()
