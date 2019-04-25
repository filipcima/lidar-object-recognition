import tensorflow as tf
import numpy as np
import os
import cv2

from SqueezeSeg import SqueezeSeg
from classifier import export_image


def main():
    squeeze_seg = SqueezeSeg()
    item_no = '004083'
    #item_no = '007362'
    #item_no = '000000'
    image, labels = get_item(item_no)
    export_image('intensity-{}-pred.png'.format(item_no), image[:, :, 0])
    export_image('gt-{}-pred.png'.format(item_no), labels)

    classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn,
        model_dir='./model/',
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": image},
        num_epochs=1,
        shuffle=False
    )

    predictions = classifier.predict(input_fn=eval_input_fn)

    idx = 0
    for p in predictions:
        cl = p['classes']
        cl[cl == 1] = 5e6
        cl[cl == 2] = 10e6
        cl[cl == 3] = 15e6
        cl[cl == 0] = 0
        export_image('cl-{}-{}.png'.format(item_no, idx), cl)
        idx += 1

    print(predictions)


def get_item(filename='000000.npy', mode='test'):
    path = os.path.join('..', 'dataset', 'lidar_2d', mode, '{}.npy'.format(filename))
    item = np.load(path).astype(np.float32)
    return item[:, :, :5], item[:, :, 5]


if __name__ == '__main__':
    main()
