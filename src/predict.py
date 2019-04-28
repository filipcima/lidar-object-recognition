import argparse
from glob import glob
import tensorflow as tf
import os
from time import time

from train import SqueezeSeg
from utils import get_item
from utils import calculate_iou
from utils import mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_number', default=os.getcwd())
    parser.add_argument('-m', '--mode', help='train|test', default='test')
    args = parser.parse_args()
    item_no = args.frame_number
    squeeze_seg = SqueezeSeg()

    data_set_paths = glob(os.path.join('..', 'dataset', 'lidar_2d', args.mode, '*.npy'))
    file_names = []
    for path in data_set_paths:
        file_names.append(os.path.splitext(path)[0].split('/')[-1])

    all_ious = {
        'unknown': [],
        'car': [],
        'pedestrian': [],
        'cyclist': []
    }

    classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn,
        model_dir='./model/',
    )
    idx = 0
    for item_no in file_names:
        if idx % 20 == 0:
            image, labels = get_item(item_no, mode=args.mode)
            # export_image('intensity-{}-pred.png'.format(item_no), image[:, :, 0])
            # export_image('gt-{}-pred.png'.format(item_no), labels)

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": image},
                num_epochs=1,
                shuffle=False
            )

            before = time()
            predictions = classifier.predict(input_fn=eval_input_fn)
            after = time()

            pred = next(predictions)['classes']
            ious = calculate_iou(labels, pred, 4)

            print(ious)
            print('Time elapsed: {}'.format((after - before) * 1000))

            for cls, iou in ious.iteritems():
                all_ious[cls].append(iou)
        idx += 1

    all_means = {cls: mean(iou) for cls, iou in all_ious.iteritems()}
    all_means['unknown'] = None

    print(all_means)


if __name__ == '__main__':
    main()
