import numpy as np
import os
import keras
from gen import read_npz
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from model import build_model_Conv2D

import tensorflow as tf
import keras.backend as K

def f2(y_true, y_pred):
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    # TN = conf_matrix.values.sum() - (FP + FN + TP)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f2_value = 5.0*precision*recall / (4.0*precision+recall)
    f1_value = 2. * precision * recall / (precision + recall)
    return precision, recall, f1_value

def cal_f2(y_true, y_pred):
    precision, recall, f1_value = tf.py_function(f2, [y_true, y_pred], [tf.float32, tf.float32, tf.float32])
    return precision, recall, f1_value

def training(input_shape, nb_classes, batch_size=32, nb_epoch=100, model_name="Conv2D_Model", double_label=True):
    #Read dataset
    if double_label:
        X_train, y_train = read_npz('train.npz', True)
        X_val, y_val = read_npz('val.npz', False)
    else:
        X_train, y_train = read_npz('train_no_double.npz', True)
        X_val, y_val = read_npz('val_no_double.npz', False)

    print(X_train.shape)
    print(X_val.shape)

    #Scale X train and X val
    # scalers = {}
    # for i in range(X_train.shape[1]):
    #     scalers[i] = StandardScaler()
    #     X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
    #
    # for i in range(X_val.shape[1]):
    #     X_val[:, i, :] = scalers[i].transform(X_val[:, i, :])

    X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_val = np.reshape(X_val, newshape=(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))

    model = build_model_Conv2D(input_shape, nb_classes=nb_classes)
    model.summary(line_length=120)

    if not os.path.exists(model_name):
        os.mkdir(model_name)
    filepath = os.path.join(model_name, model_name + ".hdf5")

    sgd = SGD(lr=0.001, momentum=0.99, nesterov=True)
    early_stopper = EarlyStopping(patience=30, mode='min', verbose=2, monitor='val_loss')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=2, save_best_only=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=1e-8, verbose=2, monitor='val_loss',
                                  mode='min')
    log_path = os.path.join(model_name, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tensorboard = TensorBoard(log_dir=log_path)

    callback = [early_stopper, checkpointer, reduce_lr, tensorboard]
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=1, callbacks=callback,
              validation_data=(X_val, y_val))

def main():
    input_shape = (12,3000,1)
    nb_classes = 9
    batch_size = 128
    model_name = 'Conv2D_Model'
    nb_epoch = 200
    double_label = True
    training(input_shape,nb_classes,batch_size, model_name=model_name, nb_epoch=nb_epoch, double_label=double_label)

if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)
    main()