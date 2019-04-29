import argparse
from glob import glob
import tensorflow as tf
import os
from time import time

from train import SqueezeSeg
from utils import get_item
from utils import calculate_iou
from utils import mean
from utils import export_image


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

    mean_iou, mean_precision, mean_recall = {}, {}, {}

    classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn,
        model_dir='./model/',
    )
    idx = 0
    for item_no in file_names:
        if idx % 50 == 0:
            print('{}/{}'.format(idx/50, len(file_names)/50))
            image, labels = get_item(item_no, mode=args.mode)
            #export_image('intensity-{}-pred.png'.format(item_no), image[:, :, 0])
            #export_image('gt-{}-pred.png'.format(item_no), labels)

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": image},
                num_epochs=1,
                shuffle=False
            )

            before = time()
            predictions = classifier.predict(input_fn=eval_input_fn)
            after = time()

            pred = next(predictions)['classes']
            ious, precisions, recalls = calculate_iou(labels, pred, 4)

            print(ious)
            print('Time elapsed: {}'.format((after - before) * 1000))

            for cls, iou in ious.iteritems():
                mean_iou[cls].append(iou)

            for cls, precision in precisions.iteritems():
                mean_precision[cls].append(precision)

            for cls, recall in recalls.iteritems():
                mean_recall[cls].append(recall)

        idx += 1

    mean_iou = {cls: mean(iou) for cls, iou in mean_iou.iteritems()}
    mean_precision = {cls: mean(precision) for cls, precision in mean_precision.iteritems()}
    mean_recall = {cls: mean(recall) for cls, recall in mean_recall.iteritems()}

    print('IoU:', mean_iou)
    print('PRECISION:', mean_precision)
    print('RECALL:', mean_recall)

    pred[pred == 1] = 5e6
    pred[pred == 2] = 10e6
    pred[pred == 3] = 15e6
    pred[pred == 0] = 0
    #export_image('predict-{}-pred.png'.format(item_no), pred)


if __name__ == '__main__':
    main()
