import tensorflow as tf


class SqueezeSeg(object):
    def __init__(self, learning_rate=0.001, num_classes=4, batch_size=4, epochs=64, log_info=True):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        if log_info:
            tf.logging.set_verbosity(tf.logging.INFO)

    def fire_module(self, input_tensor, name, filters):
        """
        Fire module as proposed in SqueezeSeg model
        :param input_tensor: Tensor [N, H, W, C]
        :param name: module name
        :param filters: array of two filters for conv2d
        :return: Tensor [N, H, W, C]
        """

        conv_1 = tf.layers.conv2d(
            inputs=input_tensor,
            filters=filters[0],
            kernel_size=[1, 1],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same',
            name=name + '_squeeze'
        )

        conv_2_1 = tf.layers.conv2d(
            inputs=conv_1,
            filters=filters[1],
            kernel_size=[3, 3],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same',
            name=name + '_expand_1_3x3'
        )

        conv_2_2 = tf.layers.conv2d(
            inputs=conv_1,
            filters=filters[1],
            kernel_size=[1, 1],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same',
            name=name + '_expand_2_1x1'
        )

        output_tensor = tf.concat([conv_2_1, conv_2_2], -1)
        return output_tensor

    def fire_deconv_module(self, input_tensor, name):
        """
        Fire deconv module as propsed in SqueezeSeg.
        [H, W, C] -> [H, 2*W, C]
        :param input_tensor:
        :param name: module name
        :return: [N, H, 2*W, C]
        """
        C = int(input_tensor.shape[-1])

        conv_1 = tf.layers.conv2d(
            inputs=input_tensor,
            filters= C /2,
            kernel_size=[1, 1],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same', name='{}_squeeze'.format(name))

        deconv_x2 = tf.layers.conv2d_transpose(
            conv_1,
            filters= C /4,
            kernel_size=[3, 3],
            strides=[1, 2],
            activation=tf.nn.relu,
            padding='same',
            name='{}_deconv_2'.format(name)
        )

        conv_2_1 = tf.layers.conv2d(
            inputs=deconv_x2,
            filters= C /2,
            kernel_size=[3, 3],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same',
            name='{}_expand_1_3x3'.format(name)
        )

        conv_2_2 = tf.layers.conv2d(
            inputs=deconv_x2,
            filters= C /2,
            kernel_size=[1, 1],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same',
            name='{}_expand_2_1x1'.format(name))

        output_tensor = tf.concat([conv_2_1, conv_2_2], -1)
        return output_tensor

    def squeeze_seg_fn(self, features, labels, mode):
        """
        Ref: https://www.tensorflow.org/tutorials/estimators/cnn
        :param features:
        :param labels:
        :param mode:
        :return: Estimator - training|evaluation|loss|predictions
        """

        input_layer = tf.reshape(features['x'], [-1, 64, 512, 5])

        tf.summary.image('input', tf.cast(tf.reshape(input_layer[:, :, :, 0], shape=(-1, 64, 512, 1)), tf.uint8))

        conv_1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=96,
            kernel_size=[3, 7],
            strides=[1, 2],
            activation=tf.nn.relu,
            padding='same',
            name='conv1'
        )

        max_pool_1 = tf.layers.max_pooling2d(
            inputs=conv_1,
            pool_size=[1, 2],
            strides=[1, 2],
            padding='same',
            name='max_pool_1'
        )

        fire_2 = self.fire_module(max_pool_1, 'fire_2', [16, 64])
        fire_3 = self.fire_module(fire_2, 'fire_3', [16, 64])

        max_pool_3 = tf.layers.max_pooling2d(
            inputs=fire_3,
            pool_size=[1, 2],
            strides=[1, 2],
            padding='same',
            name='max_pool_3'
        )

        fire_4 = self.fire_module(max_pool_3, 'fire_4', [32, 128])
        fire_5 = self.fire_module(fire_4, 'fire_5', [32, 128])

        max_pool_5 = tf.layers.max_pooling2d(
            inputs=fire_5,
            pool_size=[1, 2],
            strides=[1, 2],
            padding='same',
            name='max_pool_5'
        )

        fire_6 = self.fire_module(max_pool_5, 'fire_6', [48, 192])
        fire_7 = self.fire_module(fire_6, 'fire_7', [48, 192])
        fire_8 = self.fire_module(fire_7, 'fire_8', [64, 256])
        fire_9 = self.fire_module(fire_8, 'fire_9', [64, 256])

        fire_deconv_10 = self.fire_deconv_module(fire_9, 'fire_deconv_10')
        skip_fire_4_deconv_10 = tf.concat([fire_4, fire_deconv_10], -1)

        fire_deconv_11 = self.fire_deconv_module(skip_fire_4_deconv_10, 'fire_deconv_11')
        skip_fire_2_deconv_11 = tf.concat([fire_2, fire_deconv_11], -1)

        fire_deconv_12 = self.fire_deconv_module(skip_fire_2_deconv_11, 'fire_deconv_12')
        skip_conv_1a_deconv_12 = tf.concat([conv_1, fire_deconv_12], -1)

        fire_deconv_13 = self.fire_deconv_module(skip_conv_1a_deconv_12, 'fire_deconv_13')
        skip_conv_1b_deconv_13 = tf.concat([input_layer, fire_deconv_13], -1)

        conv_14 = tf.layers.conv2d(
            inputs=skip_conv_1b_deconv_13,
            filters=self.num_classes,
            kernel_size=[1, 1],
            strides=[1, 1],
            activation=tf.nn.relu,
            padding='same',
            name='conv_14'
        )

        logits = conv_14

        tf.summary.image('logits', tf.reshape(logits[:, :, :, 0], shape=(-1, 64, 512, 1)))

        predictions = {
            'classes': tf.argmax(input=logits, axis=-1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        predictions_classes = tf.reshape(predictions['classes'], shape=(-1, 64, 512, 1))

        tf.summary.image('predictions', tf.cast(predictions_classes, tf.float32))
        try:
            tf.summary.image('labels', tf.cast(tf.reshape(labels, shape=(-1, 64, 512, 1)), tf.float32))
        except ValueError:
            print('error')
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.num_classes)
        weights = tf.constant([0.8, 0.5, 0.3, 0.2])
        class_weights = tf.multiply(one_hot_labels, weights)

        print('One hot labels:' ,one_hot_labels)
        print('Logits:', logits)
        print('Class weights', class_weights)

        cross_entropy = tf.multiply(
            class_weights,
            tf.losses.softmax_cross_entropy(
                onehot_labels=one_hot_labels,
                logits=logits
            )
        )

        loss = tf.reduce_mean(cross_entropy)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_optimizer = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_optimizer)

        print('Prediction classes:', predictions['classes'])

        eval_metric_ops = {
            'mean_iou': tf.metrics.mean_iou(
                labels=labels, predictions=predictions['classes'],
                num_classes=self.num_classes, name='iou_metric')
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
