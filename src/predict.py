import argparse
import tensorflow as tf
import os

from train import SqueezeSeg
from utils import export_image
from utils import get_item


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('frame_number', default=os.getcwd())
    parser.add_argument('-m', '--mode', help='train|test', default='train')
    args = parser.parse_args()
    item_no = args.frame_number
    squeeze_seg = SqueezeSeg()

    image, labels = get_item(item_no, mode=args.mode)
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
        export_image('predict-{}-{}-pred.png'.format(item_no, idx), cl)
        idx += 1

    print(predictions)


if __name__ == '__main__':
    main()
