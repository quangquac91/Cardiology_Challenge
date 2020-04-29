from keras import Model
from keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten
from keras.regularizers import l2
import keras.backend as K

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def swish(x):
    return x * K.sigmoid(x)

def build_model_Conv2D(input_shape, nb_classes, weight_decay=1e-4):
    inp = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(12,6), strides=(1,1), padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay),
               kernel_initializer=CONV_KERNEL_INITIALIZER)(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(swish)(x)
    x = MaxPooling2D(pool_size=(2, 4), strides=(1, 4), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(64, kernel_size=(12, 6), strides=(1, 1), padding='same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(swish)(x)
    x = MaxPooling2D(pool_size=(2, 4), strides=(1, 4), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, kernel_size=(12, 6), strides=(1, 1), padding='same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(swish)(x)
    x = MaxPooling2D(pool_size=(2, 4), strides=(1, 4), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, kernel_size=(12, 6), strides=(1, 1), padding='same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(swish)(x)
    x = MaxPooling2D(pool_size=(2, 4), strides=(1, 4), padding='same')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, kernel_size=(12, 6), strides=(1, 1), padding='same', use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(swish)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)

    model = Model(inputs=inp, outputs=x)
    return model

def lstm_model(input_shape, nb_classes):
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(100))(inp)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    return model

if __name__ == '__main__':
    input_shape = (12,3000,1)
    model = build_model_Conv2D(input_shape,9)
    model.summary(line_length=120)

