# UCF Senior Design 2017-18
# Group 38

from model_base import BaseModel
import tflearn

NUM_LABELS = 2


class TestModel(BaseModel):
    """
        Test Model:
        Testing the feasibility of the object-oriented version of our model
    """
    # Labels: bird, not bird
    NUM_LABELS = 2

    def __init__(self, model_name, output_folder=None, hp_content=None):
        """
            Inputs:
                model_name:     (String) name of model
                output_folder   (String) where to store saved data
        """
        BaseModel.__init__(self, model_name, hp_content=hp_content)
        self.network = tflearn.layers.core.input_data(shape=[None, 32, 32, 3])
        self.network = self._build_network(self.network)

    def _build_network(self, placeholder):
        """
            Build the network.
                placeholder:    (Tensor) Input data to be given to the network
        """

        # First: Initial layer of the CNN with ReLU and max-pooling
        network = self._conv_layer(placeholder, self.hp.conv_1_nb,
                                   self.hp.conv_1_size, activation='relu',
                                   max_pooling=True)

        # Second: A group of layers with ReLU on both and max-pooling added on the second
        network = self._conv_layer(network, self.hp.conv_2_nb,
                                   self.hp.conv_2_size, activation='relu',
                                   max_pooling=False)
        network = self._conv_layer(network, self.hp.conv_2_nb,
                                   self.hp.conv_2_size, activation='relu',
                                   max_pooling=True)

        # Third: A fully-connected layer
        network = self._fully_connected_layer(network, 512, activation='relu')

        # Fourth: Dropout layer to enforce redundancy and reduce over-fitting
        network = self._dropout(network, self.hp.dropout)

        # Fifth: Another fully-connected layer with output nodes equal to NUM_LABELS
        network = self._end_softmax_layer(network, NUM_LABELS)

        # Final: Regression estimator
        network = self._regression(network, self.hp.learning_rate)

        return network
