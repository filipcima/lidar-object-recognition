import argparse
from glob import glob
import tensorflow as tf
import os
from time import time

from train import SqueezeSeg
from utils import get_item
from utils import calculate_metrics
from utils import mean
from utils import export_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_number', default=os.getcwd())
    parser.add_argument('-m', '--mode', help='train|test', default='test')
    parser.add_argument('--export-image', help='export predicted image, gt and intensity', action='store_true')
    args = parser.parse_args()
    item_no = args.frame_number
    squeeze_seg = SqueezeSeg()

    file_names = []

    mean_iou = {
        'unknown': [],
        'car': [],
        'pedestrian': [],
        'cyclist': []
    }
    mean_precision = {
        'unknown': [],
        'car': [],
        'pedestrian': [],
        'cyclist': []
    }
    mean_recall = {
        'unknown': [],
        'car': [],
        'pedestrian': [],
        'cyclist': []
    }

    if item_no == 0:
        data_set_paths = glob(os.path.join('..', 'dataset', 'lidar_2d', args.mode, '*.npy'))
        for path in data_set_paths:
            file_names.append(os.path.splitext(path)[0].split('/')[-1])
    else:
        file_names.append(item_no)

    classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn,
        model_dir='./model/',
    )
    idx = 0
    for item_no in file_names:
        print('{}/{}'.format(idx, len(file_names)))
        image, labels = get_item(item_no, mode=args.mode)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": image},
            num_epochs=1,
            shuffle=False
        )

        before = time()
        predictions = classifier.predict(input_fn=eval_input_fn)
        after = time()

        pred = next(predictions)['classes']
        ious, precisions, recalls = calculate_metrics(labels, pred, 4)

        print('IOUS: {}'.format(ious))
        print('PRECISIONS: {}'.format(precisions))
        print('RECALL: {}'.format(recalls))
        print('Time elapsed: {}'.format((after - before) * 1000))

        for cls, iou in ious.iteritems():
            mean_iou[cls].append(iou)

        for cls, precision in precisions.iteritems():
            mean_precision[cls].append(precision)

        for cls, recall in recalls.iteritems():
            mean_recall[cls].append(recall)

        if args.export_image:
            export_image('intensity-{}-pred.png'.format(item_no), image[:, :, 0])
            export_image('gt-{}-pred.png'.format(item_no), labels)
            pred[pred == 1] = 5e6
            pred[pred == 2] = 10e6
            pred[pred == 3] = 15e6
            pred[pred == 0] = 0
            export_image('predict-{}-pred.png'.format(item_no), pred)

        idx += 1

    mean_iou = {cls: mean(iou) for cls, iou in mean_iou.iteritems()}
    mean_precision = {cls: mean(precision) for cls, precision in mean_precision.iteritems()}
    mean_recall = {cls: mean(recall) for cls, recall in mean_recall.iteritems()}

    print('IoU:', mean_iou)
    print('PRECISION:', mean_precision)
    print('RECALL:', mean_recall)


if __name__ == '__main__':
    main()
