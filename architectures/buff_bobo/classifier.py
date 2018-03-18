# UCF Senior Design 2017-18
# Group 38

import os
import tflearn

OUTPUT_FOLDER = 'output/buff_bobo'

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
N = 5

class ClassificationModel(object):
    """
        A model for object classification.
    """

    def __init__(self, image_size, weights):
        self.network = self._build_network(N, image_size)
        self.model = tflearn.DNN(self.network, checkpoint_path=OUTPUT_FOLDER,
                    max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)
        self.model.load(weights, weights_only=True)

    def _build_network(self, n, image_size):
        # Define the input to the network.

        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True)

        # Real-time data augmentation.
        img_aug = tflearn.ImageAugmentation()
        img_aug.add_random_flip_leftright()

        net = tflearn.input_data(shape=[None, image_size[0], image_size[1], 3],
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

    def predict(self, image):
        """
            Wraps the TFLearn model prediction function. Returns the
            predicted probabilites in an array.
        """
        return self.model.predict(image)
