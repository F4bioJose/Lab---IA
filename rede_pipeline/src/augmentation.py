from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_augmentation_pipeline(intensity=0.1):
    """Cria um pipeline progressivo de augmentação.

    Args:
        intensity: 0.0 (sem augmentação) a 1.0 (máxima).
    """
    intensity = max(0.0, min(1.0, float(intensity)))

    return ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=int(10 * intensity),
        width_shift_range=0.05 * intensity,
        height_shift_range=0.05 * intensity,
        shear_range=0.05 * intensity,
        zoom_range=0.05 * intensity,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='reflect',
    )
