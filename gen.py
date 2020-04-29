import os
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_hea_file(root_path):
    num_hea_file = len(os.listdir(root_path)) //2
    label_data = []
    count_double = 0
    for i in range(1,num_hea_file+1):
        hea_file_path = os.path.join(root_path, 'A' + str(i).zfill(4) + '.hea')
        f = open(hea_file_path,'r')
        temp = f.readlines()
        label = temp[15].split(': ')[1][:-1]
        if ',' in label:
            label = label.split(',')
            count_double+=1
        label_data.append(label)
    print('Double label: ',count_double, 'samples')
    return label_data

def analysis(label_data):
    label_name = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
    total = 0
    for label in label_name:
        count = 0
        for l in label_data:
            if l == label:
                count+=1
        ratio = count / 6877. * 100.
        ratio = round(ratio,2)
        print('Label ', label,' has: ', count,' samples (ratio: ', ratio,'%)')
        total+=count
    print('Total (single label): ', total, 'samples')
    print('Total samples: 6877 samples')


def encoding_label(label_data):
    label_name = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
    encoding_label = []
    for l in label_data:
        if type(l) is str:
            label = np.zeros(9, dtype=int)
            label[label_name.index(l)] = 1
        else:
            label = np.zeros(9, dtype=int)
            for i in l:
                label[label_name.index(i)] = 1
        encoding_label.append(label)
    return encoding_label

def read_mat_file(root_path):
    num_mat_file = len(os.listdir(root_path)) // 2
    X_leads = []

    for i in range(1,num_mat_file+1):
        mat_file_path = os.path.join(root_path, 'A' + str(i).zfill(4) + '.mat')
        mat_contents = sio.loadmat(mat_file_path)['val']
        X_leads.append(mat_contents)
    return X_leads

def remove_double_label(X_leads, y):
    new_X, new_y = [], []
    for i in range(len(y)):
        if type(y[i]) is str:
            new_X.append(X_leads[i])
            new_y.append(y[i])
    return new_X, new_y

def extend_data(X, y, length_sample=3000):
    new_X, new_y = [], []
    for i in range(len(y)):
        length = X[i].shape[1]
        repeat = length // length_sample
        for k in range(repeat):
            temp_X = X[i][:,k*length_sample: (k+1)*length_sample]
            temp_y = y[i]
            new_X.append(temp_X)
            new_y.append(temp_y)
    return new_X, new_y

def read_npz(npz_file, istrain):
    d = np.load(npz_file)
    if istrain:
        return d['X_train'], d['y_train']
    else:
        return d['X_val'], d['y_val']

label_data = read_hea_file('Training_WFDB')
analysis(label_data)
# X_leads = read_mat_file('Training_WFDB')
# X_leads, label_data = remove_double_label(X_leads, label_data)
# y = encoding_label(label_data)
#
#
# X_train, X_val, y_train, y_val = train_test_split(X_leads, y, test_size=0.1)
# new_X_train, new_y_train = extend_data(X_train, y_train)
# new_X_val, new_y_val = extend_data(X_val, y_val)
# np.savez('train_no_double.npz', X_train=np.asarray(new_X_train), y_train=np.asarray(new_y_train))
# np.savez('val_no_double.npz', X_val=np.asarray(new_X_val), y_val=np.asarray(new_y_val))

# X_train, y_train = read_npz('train_no_double.npz', True)
# X_val, y_val = read_npz('val_no_double.npz', False)
#
# print(X_train.shape)
# print(X_val.shape)


