import cv2
import os
import numpy as np

KITTI_LIST = ('unknown', 'car', 'pedestrian', 'cyclist')


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


def calculate_iou(label, pred, num_classes=4, epsilon=1e-12):
    ious = np.zeros(num_classes)
    tps = np.zeros(num_classes)
    fns = np.zeros(num_classes)
    fps = np.zeros(num_classes)

    for cls_id in range(num_classes):
        tp = np.sum(pred[label == cls_id] == cls_id)
        fp = np.sum(label[pred == cls_id] != cls_id)
        fn = np.sum(pred[label == cls_id] != cls_id)

        ious[cls_id] = tp / (tp + fn + fp + epsilon)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn

    return {KITTI_LIST[i]: iou for i, iou in enumerate(ious) if tps[i] > 0 or fps[i] > 0 or fns[i] > 0}
