# UCF Senior Design 2017-18
# Group 38

import tflearn
import yaml


class Model(object):
    """
        This model is inspired by the SqueezeNet architecture, a model that
        achieves AlexNet-level accuracy, but with 50x less parameters.
        Official repo: https://github.com/DeepScale/SqueezeNet
    """

    def __init__(self, config):
        """
            Initialize the model.
                image_size:     (Tuple) size of image in form (width, height)
                num_classes:    (Integer) number of classes to be classified
        """
        with open(config, 'r') as f:
            self.cfg = yaml.load(f)

        self.cfg = self.cfg['model']['network']
        self.fire = self.cfg['fire']
        self.nest = self.cfg['nesterov']
        self.network = self._build_network(self.cfg['image_size'],
                                           self.cfg['num_classes'])

    def _fire_module(self, input_layer, fire_layer_id, squeeze=16, expand=64):
        """
            The Fire module from the original SqueezeNet paper.
                input_layer:    (Tensor) preceding layers of network
                fire_layer_id:  (Integer) used to differentiate layers
                squeeze:        (Integer) filter size for the 'squeeze' part of module
                expand:         (Integer) filter size for the 'expand' part of module
        """

        layer_id = 'fire' + str(fire_layer_id) + '/'

        fire = tflearn.layers.conv.conv_2d(input_layer, squeeze, 1,
                                           padding='valid', activation=self.cfg['act'],
                                           weights_init=self.cfg['weight_init'],
                                           name=layer_id + "elu_" + "squeeze1x1")

        left = tflearn.layers.conv.conv_2d(fire, expand, 1,
                                           padding='same', activation=self.cfg['act'],
                                           weights_init=self.cfg['weight_init'],
                                           name=layer_id + "elu_" + "expand1x1")

        right = tflearn.layers.conv.conv_2d(fire, expand, 3,
                                            padding='same', activation=self.cfg['act'],
                                            weights_init=self.cfg['weight_init'],
                                            name=layer_id + "elu_" + "expand3x3")

        out = tflearn.layers.merge_ops.merge([left, right], 'concat',
                                             axis=3, name=layer_id + 'merge')

        return out

    def _build_network(self, image_size, num_classes):
        """
            Build the SqueezeNet architecture.
                image_size:     (Tuple) size of image in form (width, height)
                num_classes:    (Integer) number of classes to be classified
        """
        # Augment the dataset for better training.
        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True)

        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()

        # Start the network with an input layer of custom image size.
        net = tflearn.input_data(shape=[None, image_size[0], image_size[1], 3],
                                 data_preprocessing=img_prep,
                                 data_augmentation=img_aug)

        net = tflearn.layers.conv.conv_2d(net, self.config['num_filters'],
                                          self.config['filter_size'], strides=self.cfg['strides'],
                                          padding='valid', activation=self.cfg['act'],
                                          weights_init=self.cfg['weight_init'],
                                          name='conv1')
        net = tflearn.layers.conv.max_pool_2d(net, 3, strides=2, name='pool1')

        net = self._fire_module(net, 2, self.fire['sq_2_3'], self.fire['ex_2_3'])
        net = self._fire_module(net, 3, self.fire['sq_2_3'], self.fire['ex_2_3'])
        net = tflearn.layers.conv.max_pool_2d(net, 3, strides=self.cfg['strides'], name='pool3')

        net = self._fire_module(net, 4, self.fire['sq_4_5'], self.fire['ex_4_5'])
        net = self._fire_module(net, 5, self.fire['sq_4_5'], self.fire['ex_4_5'])
        net = tflearn.layers.conv.max_pool_2d(net, 3, strides=self.cfg['strides'], name='pool5')

        net = self._fire_module(net, 6, self.fire['sq_6_7'], self.fire['ex_6_7'])
        net = self._fire_module(net, 7, self.fire['sq_6_7'], self.fire['ex_6_7'])
        net = self._fire_module(net, 8, self.fire['sq_8_9'], self.fire['ex_8_9'])
        net = self._fire_module(net, 9, self.fire['sq_8_9'], self.fire['ex_8_9'])
        net = tflearn.layers.core.dropout(net, self.cfg['dropout_prob'], name='dropout9')

        net = tflearn.layers.conv.conv_2d(net, num_classes, 1, padding='valid',
                                          activation='elu', name='conv10')
        net = tflearn.layers.conv.global_avg_pool(net)

        net = tflearn.activations.softmax(net)

        nesterov = tflearn.optimizers.Nesterov(learning_rate=self.nest['lr'],
                                               lr_decay=self.nest['lr_decay'],
                                               decay_step=self.nest['decay_step'])
        net = tflearn.layers.estimator.regression(net, optimizer=nesterov)

        return net
