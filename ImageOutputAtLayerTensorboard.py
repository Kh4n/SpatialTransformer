import tensorflow as tf

class ImageOutputAtLayerTensorboard(tf.keras.callbacks.TensorBoard):

    def __init__(self, layer_name, data, **kwargs):
        self.data = data
        self.layer_name = layer_name
        super(ImageOutputAtLayerTensorboard, self).__init__(**kwargs)
        # self.intermediate_layer_model = tf.keras.models.Model(
        #     inputs=self.model.input,
        #     outputs=self.model.get_layer(layer_name).output
        # )

    def on_epoch_begin(self, epoch, logs=None):
        self.intermediate_layer_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.layer_name).output
        )
        with self._get_writer(self._train_run_name).as_default():
            tf.summary.image("Test", self.intermediate_layer_model.predict(self.data), step=epoch)
        super().on_epoch_begin(epoch, logs)