# UCF Senior Design 2017-18
# Group 38

import json
import os
import tflearn
import time

CHECKPOINT_DIR = "checkpoints"
CONFIG_DIR = "config"
HYPERPARAMETER_FILENAME = "hyperparameters.json"
OUTPUT_DIR = "output"


class Hyperparameters(object):
    """
        This class will store the hyperparameters of the network
    """
    def __init__(self):
        super(Hyperparameters, self).__init__()
        # This list will store the hyperparameters
        self.param_list = []

    def set_hp(self, hp):
        for key in hp:
            self.param_list.append(key)
            setattr(self, key, hp[key])


class BaseModel(object):
    """
        This is the base model from which all other models will be instantiated
    """
    def __init__(self, model_name, hp_filename=None, hp_content=None):
        """
            Initialize the base model
            inputs:
                model_name:        (String)      name of model
                hp_filename:       (String)      name of hyperparameter file
                hp_content:        (JSON object) keys and values of parameters
        """
        super(BaseModel, self).__init__()

        self.name = model_name
        self.model_name = model_name

        # Get the current folder of the exectuable
        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        # We can save outputs of different models to different folders
        self.output_dir = os.path.join(self.current_dir, OUTPUT_DIR)

        self.checkpoint_dir = os.path.join(self.output_dir, CHECKPOINT_DIR)

        # The config folder will hold any specific model configurations
        self.config_dir = os.path.join(self.current_dir, CONFIG_DIR)
        if hp_filename is None:
            self.hp_path = os.path.join(self.config_dir, HYPERPARAMETER_FILENAME)
        else:
            self.hp_path = os.path.join(self.config_dir, hp_filename)

        # Load hyperparameter content
        if hp_content is None:
            self.hp_json = json.loads(self.hyparam_path)
        else:
            # This should be called if an already deserialzed JSON is passed
            self.hp_json = hp_content
        self.hp = Hyperparameters()
        self.hp.set_hp(self.hp_json)
        self._set_hyperparameters_name()
        self._set_names()

    def _conv_layer(self, incoming, nb_filter, filter_size, activation,
                    padding='same', strides=[1, 1, 1, 1], max_pooling=False,
                    maxpool_ksize=[1, 2, 2, 1], maxpool_stride=[1, 2, 2, 1], bias=True):
        """
            Create a convolutional layer w/ optional activation and/or max pooling
            inputs:
                incoming:       (Tensor)  incoming model layers
                padding:        (String)  same/valid: SAME will pad the input
                                in order for the filter to complete another full
                                operation, VALID will instead drop the
                                (right/bottom)-most columns
                activation:     (String)  enable/disable activation
                max_pooling:    (Boolean) enable max pooling
                maxpool_ksize:  (Vector)  max pooling filter size
                maxpool_stride: (Vector)  how much the max pooling kernel travels
        """

        # Attach the (padded?) convolutional layer to the rest of the model
        # print("incoming shape: ", incoming.get_shape().as_list())
        # conv_layer = tflearn.layers.conv.conv_2d(incoming, nb_filter,
        #                                         filter_size, strides, padding,
        #                                         bias)
        # print("conv_layer shape: ", conv_layer.get_shape().as_list())

        # Add activation and/or max-pooling
        if activation is None:
            conv_layer = tflearn.layers.conv.conv_2d(incoming, nb_filter,
                                                     filter_size, strides,
                                                     padding, bias)
        else:
            conv_layer = tflearn.layers.conv.conv_2d(incoming, nb_filter,
                                                     filter_size, strides,
                                                     padding, activation, bias)

        if max_pooling:
            conv_layer = tflearn.layers.conv.max_pool_2d(conv_layer, maxpool_ksize,
                                                         maxpool_stride)

        return conv_layer

    def _dropout(self, incoming, keep_prob):
        """
            Create a dropout layer that encourages the network to more
            redundant by randomly selecting input elements (neurons) and
            setting their output to zero, effectively "dropping" them
                incoming:       (Tensor)  incoming model layers
                keep_prob:      (Float)   probabiltity of a neuron to be disabled
        """

        dropout = tflearn.layers.core.dropout(incoming, keep_prob)

        return dropout

    def _end_softmax_layer(self, incoming, num_labels, activation='softmax'):
        """
            Creates the final layer of the network, which is typically a fully
            connected layer with a softmax function that outputs the
            probability totals of each of the labels in the network
                incoming:       (Tensor)  incoming model layers
                num_labels:     (Integer) number of labels
                activation:     (String)  softmax function
        """

        end_layer = tflearn.layers.fully_connected(incoming, num_labels, activation)
        return end_layer

    def _fully_connected_layer(self, incoming, n_units, activation='relu', bias=True):
        """
            Create a fully connected layer w/ optional activation and/or softmax
                incoming:       (Tensor)  incoming model layers
                n_units:        (Integer) size of input
                activation:     (String)  set the activation function (default: relu)
                bias:           (Boolean) enable/disable bias
        """

        fully_connected = tflearn.layers.fully_connected(incoming, n_units, activation, bias)

        return fully_connected

    def _regression(self, incoming, learning_rate, optimizer='adam',
                    loss='categorical_crossentropy'):
        """
            Create a estmator layer that applies a regression to the layer.
                incoming:       (Tensor) incoming model layers
                optimizer:      (String) adjusts the learning rate
                loss:           (String) loss function to be used for regression
                learning_rate:  (Float)  the change interval for parameters
        """

        regression = tflearn.layers.estimator.regression(incoming, optimizer=optimizer,
                                                         loss=loss, learning_rate=learning_rate)

        return regression

    def _set_hyperparameters_name(self):
        """
            Convert hyperparameters dict to a string
            This string will be used to set the models names
        """
        # Generate a little name for each hyperparameters
        hyperparameters_names = [("".join([p[0] for p in hp.split("_")]), getattr(self.hp, hp))
                                 for hp in self.hp.param_list]
        self.hyperparameters_name = ""
        for index_hyperparameter, hyperparameter in enumerate(hyperparameters_names):
            short_name, value = hyperparameter
            prepend = "" if index_hyperparameter == 0 else "_"
            self.hyperparameters_name += "%s%s_%s" % (prepend, short_name, value)

    def _set_names(self):
        """
            Set all model names
        """
        name_time = "%s--%s" % (self.model_name, time.time())
        # model_name is used to set the ckpt name
        self.model_name = "%s--%s" % (self.hyperparameters_name, name_time)
        # sub_train_log_name is used to set the name of the training part in tensorboard
        self.sub_train_log_name = "%s-train--%s" % (self.hyperparameters_name, name_time)
        # sub_test_log_name is used to set the name of the testing part in tensorboard
        self.sub_test_log_name = "%s-test--%s" % (self.hyperparameters_name, name_time)
