import tensorflow as tf
import numpy as np
import os
import cv2

from SqueezeSeg import SqueezeSeg, get_data


def main():
    squeeze_seg = SqueezeSeg()
    item_no = '000000'
    item = get_item(item_no)

    data = np.array(get_data('test')).astype(np.float32)

    export_image('intensity-{}.png'.format(item_no), data[0, :, :, 0])
    export_image('x-{}.png'.format(item_no), data[0, :, :, 1])
    export_image('y-{}.png'.format(item_no), data[0, :, :, 2])
    export_image('z-{}.png'.format(item_no), data[0, :, :, 3])
    export_image('distance-{}.png'.format(item_no), data[0, :, :, 4])
    export_image('gt-{}.png'.format(item_no), data[0, :, :, 5])

    classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn,
        model_dir='./model/',
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data[:, :, :, :5]},
        y=data[:, :, :, 5],
        num_epochs=1,
        shuffle=False
    )

    print('Started eval...')
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print('Eval done.')


def export_image(name, image):
    try:
        image = cv2.normalize(image[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    except IndexError as e:
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite(name, image)


def get_item(filename='000000.npy', mode='test'):
    path = os.path.join('..', 'dataset', 'lidar_2d', mode, '{}.npy'.format(filename))
    item = np.load(path).astype(np.float32)
    return [item[:, :, :5], item[:, :, 5]]


if __name__ == '__main__':
    main()
