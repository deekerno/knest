# UCF Senior Design 2017-18
# Group 38

import tflearn

class BuffBobo(object):
    """
        The "buffed" version of our original classification network,
        affectionately named Bobo. The structure of this network is
        inspired by ResNet.
    """

    def __init__(self, n, img_aug, img_prep):
        self.network = self._build_network(n, img_aug, img_prep)

    
    def _build_network(self, n, img_aug, img_prep):
        # Define the input to the network.
        net = tflearn.input_data(shape=[None, 112, 112, 3],
                                 data_preprocessing=img_prep,
                                 data_augmentation=img_aug)

        # Start with a normal convolutional layer.
        net = tflearn.conv_2d(net, 64, 3, regularizer='L2', weight_decay=0.0001)

        # Since this is a ResNet with <50 layers, we'll use regular residual blocks;
        # otherwise, we'd use residual bottleneck blocks instead.
        net = tflearn.residual_block(net, n, 64)
        net = tflearn.residual_block(net, 1, 128, downsample=True)
        net = tflearn.residual_block(net, n-1, 128)
        net = tflearn.residual_block(net, 1, 256, downsample=True)
        net = tflearn.residual_block(net, n-1, 256)

        # Perform batch normalization.
        net = tflearn.batch_normalization(net)

        # Activation at the end of the network pre-FC.
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)


        net = tflearn.fully_connected(net, 2, activation='softmax')
        mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        net = tflearn.regression(net, optimizer=mom,
                                 loss='categorical_crossentropy')

        return net
