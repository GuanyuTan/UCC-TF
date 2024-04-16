import tensorflow as tf
import numpy as np


class ResBlockZeroPadding(tf.keras.Model):
    def __init__(self, filters, downsample=False, upsample=False, first_block=False):
        super(ResBlockZeroPadding, self).__init__()
        assert not (
            upsample and downsample), "Only set upsample or downsample as true"
        self.upsample = upsample
        self.downsample = downsample
        self.first_block = first_block
        self.filters = filters
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(3, 3),
            padding='same',
            strides=(2, 2) if downsample else (1, 1)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(3, 3),
            padding='same',
            strides=(1, 1)
        )
        # 1x1 Convolution to help with downsample,
        self.skip_conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            padding='same',
            strides=(1, 1)
        )
        self.upsampling = tf.keras.layers.UpSampling2D()
        self.activation = tf.keras.activations.relu

    def call(self, input_tensor):
        if self.first_block:
            input_tensor = self.activation(input_tensor)
            if self.upsample:
                input_tensor = self.upsampling(input_tensor)
        x = self.conv1(input_tensor)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        if input_tensor.shape != x.shape:
            input_tensor = self.skip_conv(x)
        x += input_tensor
        return x


class WideResidualBlock(tf.keras.Model):
    def __init__(self, filters, n_layers, downsample=False, upsample=False):
        super(WideResidualBlock, self).__init__()
        self.blocks = tf.keras.Sequential(
            *[
                ResBlockZeroPadding(
                    filters=filters,
                    first_block=(i == 0),
                    downsample=downsample,
                    upsample=upsample
                )
                for i in range(n_layers)
            ]
        )

    def call(self, x):
        return self.blocks(x)


class UCCModel(tf.keras.Model):
    def __init__(self, cfg):
        super(UCCModel, self).__init__()
        model_cfg = cfg.model
        args = cfg.args
        self.num_instances = args.num_instances
        self.image_size = model_cfg.image_size
        self.num_features = model_cfg.encoder.num_features
        self.num_channels = model_cfg.num_channels
        self.num_classes = args.ucc_end-args.ucc_start+1
        self.batch_size = args.num_samples_per_class*self.num_classes
        self.alpha = model_cfg.loss.alpha
        self.sigma = model_cfg.kde_model.sigma
        self.num_nodes = model_cfg.kde_model.num_bins
        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.ReLU()
        self.decoder_reshape_size = model_cfg.decoder.reshape_size
        self.decoder_linear_size = model_cfg.decoder.linear_size
        self.encoder_linear_size = model_cfg.encoder.flatten_size
        self.dropout_rate = model_cfg.classification_model.dropout_rate
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.Input(
                    (self.image_size, self.image_size, self.num_channels)),
                tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
                WideResidualBlock(filters=32, n_layers=1),
                WideResidualBlock(filters=64, n_layers=1,  downsample=True),
                WideResidualBlock(filters=128, n_layers=1,  downsample=True),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape((self.encoder_linear_size,)),
                tf.keras.layers.Dense(self.num_features, activation='sigmoid')
            ]
        )
        if self.alpha == 1:
            self.decoder = None
        else:
            self.decoder = tf.keras.Sequential(
                [
                    tf.keras.Input((self.num_features,)),
                    tf.keras.layers.Dense(
                        self.decoder_linear_size, activation='relu'),
                    tf.keras.layers.Reshape(self.decoder_reshape_size),
                    WideResidualBlock(filters=64, n_layers=1, upsample=True, ),
                    WideResidualBlock(filters=32, n_layers=1, upsample=True, ),
                    WideResidualBlock(filters=16, n_layers=1, ),
                    self.relu,
                    tf.keras.layers.Conv2D(
                        filters=self.num_channels, kernel_size=(3, 3), padding='same')
                ]
            )
        self.ucc_classifier = tf.keras.Sequential(
            [
                tf.keras.Input((self.num_features, self.num_nodes)),
                tf.keras.layers.Reshape([110]),
                tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            ]
        )

    def kde(self, data, num_nodes, sigma):
        # our input data shape is [batch_size, num_instances, num_features]
        # batchsize x bins
        # we want each linspace to represent a distribution for each feature of each instance. Expected shape = [batch_size, num_instance, num_nodes]
        k_sample_points = tf.constant(
            np.tile(np.linspace(0, 1, num=num_nodes), [self.num_instances, 1]).astype(np.float32))
        k_alpha = tf.constant(
            np.array(1/np.sqrt(2*np.pi*np.square(sigma))).astype(np.float32))
        k_beta = tf.constant(
            np.array(-1/(2*np.square(sigma))).astype(np.float32))
        out = []
        # for concatenating across each feature point
        for i in range(self.num_features):
            # For each feature point
            # Reshape for broadcasting
            temp = tf.reshape(
                data[:, :, i], (self.batch_size, self.num_instances, 1))
            # get x-x_0 values into a grid
            k_diff = k_sample_points - tf.tile(temp, [1, 1, num_nodes])
            diff_sq = tf.square(k_diff)
            k_result = k_alpha * tf.exp(k_beta*diff_sq)
            # add all the feature values across instances. Expected shape = [batch_size, num_nodes]
            k_out_unnormalized = tf.reduce_sum(k_result, axis=1)
            k_norm_coeff = tf.reshape(tf.reduce_sum(
                k_out_unnormalized, axis=1), (-1, 1))
            k_out = k_out_unnormalized / \
                tf.tile(k_norm_coeff, [1, k_out_unnormalized.shape[1]])
            out.append(tf.reshape(k_out, [self.batch_size, 1, num_nodes]))
        # Expected output shape =[batch_size,num_features, num_nodes]
        concat_out = tf.concat(out, axis=1)
        return concat_out

    def call(self, inputs):

        input_list = list()
        feature_list = list()
        ae_output_list = list()
        if self.decoder:
            for i in range(self.batch_size):
                temp_input = inputs[i, :, :, :, :]
                temp_feature = self.encoder(temp_input)
                temp_feature = tf.reshape(
                    temp_feature,
                    (1, self.num_instances, self.num_features))

                temp_ae_output = self.decoder(self.encoder(temp_input))
                temp_ae_output = tf.reshape(
                    temp_ae_output, (1, self.num_instances, self.image_size, self.image_size, self.num_channels))
                input_list.append(temp_input)
                feature_list.append(temp_feature)
                ae_output_list.append(temp_ae_output)
            feature_concatenated = tf.concat(feature_list, axis=0)
            feature_distributions = self.kde(feature_concatenated, self.num_nodes, self.sigma)
            output = self.ucc_classifier(feature_distributions)
            ae_output = tf.concat(ae_output_list, axis=0)

            return output, ae_output
        else:
            for i in range(self.batch_size):
                temp_input = inputs[i, :, :, :, :]
                temp_feature = self.encoder(temp_input)
                temp_feature = tf.reshape(
                    temp_feature, (1, self.num_instances, self.num_features))

                input_list.append(temp_input)
                feature_list.append(temp_feature)
            feature_concatenated = tf.concat(feature_list, axis=0)
            feature_distributions = self.kde(feature_concatenated, self.num_nodes, self.sigma)
            output = self.ucc_classifier(feature_distributions)
            return output

    def loss(self, inputs=None, labels=None, output=None, reconstruction=None):
        if self.alpha == 1:
            assert not isinstance(output, type(
                None)), "Output classes must be provided"
            cce = tf.keras.losses.CategoricalCrossentropy()
            return cce(labels, output)
        elif self.alpha == 0:
            mse = tf.keras.losses.MeanSquaredError()
            return mse(reconstruction, inputs)
        else:

            assert not isinstance(output, type(
                None)), "Output classes must be provided"
            assert not isinstance(labels, type(
                None)), "Labels must be provided"
            assert not isinstance(inputs, type(
                None)), "Inputs must be provided"
            assert not isinstance(reconstruction, type(
                None)), "Reconstructed input must be provided"
            cce = tf.keras.losses.CategoricalCrossentropy()
            mse = tf.keras.losses.MeanSquaredError()
            return cce(labels, output), mse(reconstruction, inputs)

    def reconstruct_image(self, features):
        assert self.decoder != None, "Model does not have a decoder"
        return self.decoder(features)

    def ucc_model(self, inputs):
        feature_list = list()
        for i in range(self.batch_size):
            temp_input = inputs[i, :, :, :, :]
            temp_feature = self.encoder(temp_input)
            temp_feature = tf.reshape(
                temp_feature, (1, self.num_instances, self.num_features))
            feature_list.append(temp_feature)

        feature_concatenated = tf.concat(feature_list, axis=0)
        feature_distributions = self.kde(
            feature_concatenated, self.num_nodes, self.sigma)
        output = self.ucc_classifier(feature_distributions)
        return output

    def extract_features(self, inputs):
        steps = self.batch_size*self.num_instances
        length = inputs.shape[0]
        remainder = length % steps
        feature_list = []
        for i in range(np.ceil(length/steps).astype(np.int32)):
            start = i*steps
            if i+1 > np.ceil(length/steps):
                stop = start + remainder
            else:
                stop = (i+1)*steps
            input_array = inputs[start:stop, :, :, :]
            feature_list.append(self.encoder(input_array))
        feature_concatenated = tf.concat(feature_list, axis=0)
        return feature_concatenated
