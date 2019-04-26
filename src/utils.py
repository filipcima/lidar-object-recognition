import cv2
import os
import numpy as np


def get_item(frame_number='000000', mode='test'):
    path = os.path.join('..', 'dataset', 'lidar_2d', mode, '{}.npy'.format(frame_number))
    item = np.load(path).astype(np.float32)
    return item[:, :, :5], item[:, :, 5]


def export_image(name, image):
    try:
        image = cv2.normalize(image[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    except IndexError as e:
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite(name, image)
