import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.quantization import Q17ClipConstraint


def build_tiny_cnn(hp, num_classes, cfg):
    weight_constraint = None
    if cfg.ENABLE_HARD_WEIGHT_CONSTRAINT_DURING_TRAIN:
        weight_constraint = Q17ClipConstraint(cfg.QUANT_FRAC_BITS)

    model = keras.Sequential()
    model.add(layers.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS)))

    # 1. Camada Conv2D
    model.add(layers.Conv2D(
        filters=4,
        kernel_size=(3, 3),
        activation='relu',
        padding='valid',
        kernel_initializer='he_normal',
        kernel_constraint=weight_constraint,
        bias_constraint=weight_constraint,
        name='conv2d_hardware'
    ))

    # 2. Camada Pooling
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    # 3. Dropout para regularização do treino
    hp_dropout = hp.Choice('dropout', values=cfg.SEARCH_DROPOUT_VALUES)
    model.add(layers.Dropout(rate=hp_dropout))

    # CAMADA DENSE OCULTA REMOVIDA DAQUI!

    # 4. Camada Dense (FC) de Classificação
    model.add(layers.Dense(
        num_classes,
        activation='softmax',
        kernel_constraint=weight_constraint,
        bias_constraint=weight_constraint,
        name='dense_multiclass'
    ))

    hp_lr = hp.Choice('learning_rate', values=cfg.SEARCH_LR_VALUES)
    hp_optimizer = hp.Choice('optimizer', values=cfg.SEARCH_OPTIMIZERS)

    if hp_optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=hp_lr)
    elif hp_optimizer == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=hp_lr)
    else:
        raise ValueError(f"Otimizador nao suportado: {hp_optimizer}")

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')
        ]
    )

    return model