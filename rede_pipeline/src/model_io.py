import tensorflow as tf
from src.quantization import Q17ClipConstraint


def load_tinycnn_model(model_path, compile_model=False):
    """Carrega modelo com custom_objects usados pela Tiny-CNN."""
    return tf.keras.models.load_model(
        str(model_path),
        custom_objects={"Q17ClipConstraint": Q17ClipConstraint},
        compile=compile_model,
    )

