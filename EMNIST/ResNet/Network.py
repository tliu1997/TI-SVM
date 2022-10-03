import tensorflow as tf
# import torch

"""This script defines the network.
"""


class ResNet(object):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, inputs, training):
        """Classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        outputs = self._start_layer(inputs, training)
        if self.configs['resnet_version'] == 1:
            block_fn = self._standard_block_v1
        else:
            block_fn = self._bottleneck_block_v2

        for i in range(3):
            filters = self.configs['first_num_filters'] * (2 ** i)
            strides = 1 if i == 0 else 2
            outputs = self._stack_layer(outputs, filters, block_fn, strides, training)

        outputs = self._output_layer(outputs, training)
        return outputs

    ################################################################################
    # Blocks building the network
    ################################################################################
    def _batch_norm_relu(self, inputs, training):
        """Perform batch normalization then relu."""

        outputs = tf.layers.batch_normalization(inputs, training=training)
        outputs = tf.nn.relu(outputs)

        return outputs

    def _start_layer(self, inputs, training):
        """Implement the start layer.

        Args:
            inputs: A Tensor of shape [<batch_size>, 32, 32, 3].
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            outputs: A Tensor of shape [<batch_size>, 32, 32, self.first_num_filters].
        """
        # initial conv1
        outputs = tf.layers.conv2d(inputs, self.configs['first_num_filters'], 3, 1, padding='same')

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.configs['resnet_version'] == 1:
            outputs = self._batch_norm_relu(outputs, training)

        return outputs

    def _output_layer(self, inputs, training):
        """Implement the output layer.

        Args:
            inputs: A Tensor of shape [<batch_size>, 8, 8, channels].
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            outputs: A logits Tensor of shape [<batch_size>, self.num_classes].
        """

        # Only apply the BN and ReLU for model that does pre_activation in each
        # bottleneck block, e.g. resnet V2.
        if self.configs['resnet_version'] == 2:
            inputs = self._batch_norm_relu(inputs, training)

        # average pooling + FC layers
        outputs = tf.reduce_mean(inputs, axis=[1, 2])
        outputs = tf.layers.dense(outputs, self.configs['num_classes'])

        return outputs

    def _stack_layer(self, inputs, filters, block_fn, strides, training):
        """Creates one stack of standard blocks or bottleneck blocks.

        Args:
            inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
            filters: A positive integer. The number of filters for the first
                convolution in a block.
            block_fn: 'standard_block' or 'bottleneck_block'.
            strides: A positive integer. The stride to use for the first block. If
                greater than 1, this layer will ultimately downsample the input.
            training: A boolean. Used by operations that work differently
                in training and testing phases.

        Returns:
            outputs: The output tensor of the block layer.
        """

        filters_out = filters * 4 if self.configs['resnet_version'] == 2 else filters

        def projection_shortcut(inputs):
            # 1 * 1 convolution
            return tf.layers.conv2d(inputs, inputs.shape.as_list()[3] * 2, kernel_size=1, strides=2, padding='same')

        # Only the first block per stack_layer uses projection_shortcut
        outputs = inputs

        for idx in range(self.configs['resnet_size']):
            if idx == 0:
                if block_fn == self._standard_block_v1:
                    if strides == 2:
                        outputs = self._standard_block_v1(outputs, filters_out, training, projection_shortcut, strides)
                    else:
                        outputs = self._standard_block_v1(outputs, filters_out, training, None, 1)
                else:
                    outputs = self._bottleneck_block_v2(outputs, filters_out, training, projection_shortcut, strides)
            else:
                if block_fn == self._standard_block_v1:
                    outputs = self._standard_block_v1(outputs, filters_out, training, None, 1)
                else:
                    outputs = self._bottleneck_block_v2(outputs, filters_out, training, None, 1)

        return outputs

    def _standard_block_v1(self, inputs, filters, training, projection_shortcut, strides):
        """Creates a standard residual block for ResNet v1.

        Args:
            inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
            filters: A positive integer. The number of filters for the first
                convolution.
            training: A boolean. Used by operations that work differently
                in training and testing phases.
            projection_shortcut: The function to use for projection shortcuts
                  (typically a 1x1 convolution when downsampling the input).
            strides: A positive integer. The stride to use for the block. If
                greater than 1, this block will ultimately downsample the input.

        Returns:
            outputs: The output tensor of the block layer.
        """

        shortcut = inputs

        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        outputs = tf.layers.conv2d(inputs, filters, 3, strides, 'same')
        outputs = self._batch_norm_relu(outputs, training=training)
        outputs = tf.layers.conv2d(outputs, filters, 3, 1, 'same')
        outputs = tf.layers.batch_normalization(outputs, training=training)
        outputs = tf.nn.relu(outputs + shortcut)

        return outputs

    def _bottleneck_block_v2(self, inputs, filters, training, projection_shortcut, strides):
        """Creates a bottleneck block for ResNet v2.

        Args:
            inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
            filters: A positive integer. The number of filters for the first
                convolution. NOTE: filters_out will be 4xfilters.
            training: A boolean. Used by operations that work differently
                in training and testing phases.
            projection_shortcut: The function to use for projection shortcuts
                  (typically a 1x1 convolution when downsampling the input).
            strides: A positive integer. The stride to use for the block. If
                greater than 1, this block will ultimately downsample the input.

        Returns:
            outputs: The output tensor of the block layer.
        """
        if inputs.shape[1] % 2 != 0:
            inputs = tf.pad(inputs, [[0,0],[0,1],[0,1],[0,0]], "CONSTANT")
        shortcut = self._batch_norm_relu(inputs, training=training)

        if projection_shortcut is not None:
            shortcut = tf.layers.conv2d(shortcut, filters, 1, strides, 'same')

        outputs = self._batch_norm_relu(inputs, training=training)
        outputs = tf.layers.conv2d(outputs, filters / 4, 1, 1, 'same')
        outputs = self._batch_norm_relu(outputs, training)
        outputs = tf.layers.conv2d(outputs, filters / 4, 3, strides, 'same')
        outputs = self._batch_norm_relu(outputs, training)
        outputs = tf.layers.conv2d(outputs, filters, 1, 1, 'same')
        outputs = outputs + shortcut

        return outputs