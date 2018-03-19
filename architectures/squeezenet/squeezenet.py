# UCF Senior Design 2017-18
# Group 38

import tflearn

class Model(object):
    """
        This model is inspired by the SqueezeNet architecture, a model that
        achieves AlexNet-level accuracy, but with 50x less parameters.
        Official repo: https://github.com/DeepScale/SqueezeNet
    """

    def __init__(self, image_size, num_classes):
        """
            Initialize the model.
                image_size:     (Tuple) size of image in form (width, height)
                num_classes:    (Integer) number of classes to be classified
        """
        
        self.network = self._build_network(image_size, num_classes)

    
    def _fire_module(self, input_layer, fire_layer_id, squeeze=16, expand=64):
        """
            The Fire module from the original SqueezeNet paper.
                input_layer:    (Tensor) preceding layers of network
                fire_layer_id:  (Integer) used to differentiate layers
                squeeze:        (Integer) filter size for the 'squeeze' part of module
                expand:         (Integer) filter size for the 'expand' part of module
        """

        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        elu = "elu_"

        layer_id = 'fire' + str(fire_layer_id) + '/'

        fire = tflearn.layers.conv.conv_2d(input_layer, squeeze, 1,
                                           padding='valid', activation='elu',
                                           weights_init='xavier',
                                           name=layer_id + "elu_" + "squeeze1x1")

        left = tflearn.layers.conv.conv_2d(input_layer, expand, 1,
                                           padding='valid', activation='elu',
                                           weights_init='xavier',
                                           name=layer_id + "elu_" + "expand1x1")

        right = tflearn.layers.conv.conv_2d(input_layer, expand, 3,
                                           padding='same', activation='elu',
                                           weights_init='xavier',
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

        net = tflearn.layers.conv.conv_2d(net, 64, 3, strides=2,
                                          padding='valid', activation='elu',
                                          weights_init='xavier',
                                          name='conv1')
        net = tflearn.layers.conv.max_pool_2d(net, 3, strides=2, name='pool1')

        net = self._fire_module(net, fire_layer_id=2, squeeze=16, expand=64)
        net = self._fire_module(net, fire_layer_id=3, squeeze=16, expand=64)
        net = tflearn.layers.conv.max_pool_2d(net, 3, strides=2, name='pool3')

        net = self._fire_module(net, fire_layer_id=4, squeeze=32, expand=128)
        net = self._fire_module(net, fire_layer_id=5, squeeze=32, expand=128)
        net = tflearn.layers.conv.max_pool_2d(net, 3, strides=2, name='pool5')

        net = self._fire_module(net, fire_layer_id=6, squeeze=48, expand=192)
        net = self._fire_module(net, fire_layer_id=7, squeeze=48, expand=192)
        net = self._fire_module(net, fire_layer_id=8, squeeze=64, expand=256)
        net = self._fire_module(net, fire_layer_id=9, squeeze=64, expand=256)
        net = tflearn.layers.core.dropout(net, 0.5, name='dropout9')

        net = tflearn.layers.conv.conv_2d(net, num_classes, 1, padding='valid',
                                          activation='elu', name='conv10')
        net = tflearn.layers.conv.global_avg_pool(net)

        net = tflearn.activations.softmax(net)

        nesterov = tflearn.optimizers.Nesterov(learning_rate=0.001, lr_decay=0.96, decay_step=100)
        net = tflearn.layers.estimator.regression(net, optimizer=nesterov)
        
        return net
