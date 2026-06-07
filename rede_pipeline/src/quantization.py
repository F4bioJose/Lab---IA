import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Lab1")
class Q17ClipConstraint(tf.keras.constraints.Constraint):
    """Mantem pesos no intervalo representavel por Q1.7."""

    def __init__(self, frac_bits=7):
        self.frac_bits = frac_bits
        self.min_val = -1.0
        self.max_val = (2 ** frac_bits - 1) / (2 ** frac_bits)

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)

    def get_config(self):
        return {"frac_bits": self.frac_bits}


class Q17WeightQuantizationCallback(tf.keras.callbacks.Callback):
    """Quantiza pesos para Q1.7 durante o treino para reduzir gap treino->FPGA."""

    def __init__(self, frac_bits=7, start_epoch=0, frequency=1):
        super().__init__()
        self.frac_bits = frac_bits
        self.start_epoch = start_epoch
        self.frequency = max(1, int(frequency))
        self.scale = float(2 ** frac_bits)
        self.min_val = -1.0
        self.max_val = (2 ** frac_bits - 1) / (2 ** frac_bits)

    def __deepcopy__(self, memo):
        return Q17WeightQuantizationCallback(
            frac_bits=self.frac_bits,
            start_epoch=self.start_epoch,
            frequency=self.frequency,
        )

    def _quantize_var(self, var):
        clipped = tf.clip_by_value(var, self.min_val, self.max_val)
        quantized = tf.round(clipped * self.scale) / self.scale
        var.assign(quantized)

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        if ((epoch - self.start_epoch) % self.frequency) != 0:
            return
        for var in self.model.trainable_variables:
            self._quantize_var(var)
