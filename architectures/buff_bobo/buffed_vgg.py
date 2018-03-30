# UCF Senior Design 2017-18
# Group 38

import tflearn

class BuffBobo(object):
    """
        The "buffed" version of our original classification network,
        affectionately named Bobo. The structure of this network is
        inspired by VGG.
    """

    def __init__(self, n, img_aug, img_prep):
        self.network = self._build_network(n, img_aug, img_prep)

    
    def _build_network(self, n, img_aug, img_prep):
        # Define the input to the network.
        network = tflearn.input_data(shape=[None, 448, 448, 3],
                                 data_preprocessing=img_prep,
                                 data_augmentation=img_aug)

        network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.core.fully_connected(network, 4096, activation='relu')
        network = tflearn.layers.core.dropout(network, 0.5)
        network = tflearn.layers.core.fully_connected(network, 4096, activation='relu')
        network = tflearn.layers.core.dropout(network, 0.5)
        network = tflearn.layers.core.fully_connected(network, 2, activation='softmax')

        network = tflearn.layers.estimator.regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)
        
        return network
