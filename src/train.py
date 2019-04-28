from __future__ import absolute_import, print_function

import argparse
import os
import tensorflow as tf
import numpy as np
from glob import glob

from squeeze_seg import SqueezeSeg


def get_data(mode='train'):
    print('Retrieving {}ing data...'.format(mode))
    data_set_paths = glob(os.path.join('..', 'dataset', 'lidar_2d', mode, '*.npy'))

    data_count = len(data_set_paths)
    current_count = 0
    input_tensors = []

    for path in data_set_paths:
        if current_count % 100 == 0:
            print('Processed: {} %'.format(current_count * 1.0 / data_count * 100.0))

        input_tensors.append(np.load(path).astype(np.float32))
        current_count += 1

    print('{}ing data loaded.'.format(mode))
    return input_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=4)
    parser.add_argument('-e', '--epochs', help='epochs count', type=int, default=64)
    parser.add_argument('-m', '--mode', help='train|eval', default='train')
    parser.add_argument('--verbose', help='make tensorflow verbose', action='store_true')
    args = parser.parse_args()

    squeeze_seg = SqueezeSeg(batch_size=args.batch_size, epochs=args.epochs, log_info=args.verbose)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(session_config=config)
    classifier = tf.estimator.Estimator(
        model_fn=squeeze_seg.squeeze_seg_fn,
        model_dir='./model/',
        config=run_config
    )

    whole_data = np.array(get_data(args.mode if args.mode == 'train' else 'test')).astype(np.float32)
    data = whole_data[:, :, :, :5]
    labels = whole_data[:, :, :, 5]

    if args.mode == 'train':
        tensors_to_log = {
            "probabilities": "softmax_tensor",
        }

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            y=labels,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            shuffle=True
        )

        print('Started training...')
        classifier.train(
            input_fn=train_input_fn,
            steps=2000,
            hooks=[logging_hook]
        )
        print('Training done.')

    if args.mode == 'eval':
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            y=labels,
            num_epochs=1,
            shuffle=False,
            batch_size=args.batch_size
        )

        print('Started eval...')
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print('Eval done.')

        print(eval_results)


if __name__ == '__main__':
    main()
