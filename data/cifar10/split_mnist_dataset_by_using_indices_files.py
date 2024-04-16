import numpy as np

import keras
from keras.datasets import cifar10

(original_x_train, original_y_train), (x_test, y_test) = cifar10.load_data()

print('original_x_train shape:{}'.format(original_x_train.shape))
print('x_test shape:{}'.format(x_test.shape))

# read splitting indices files
training_split_indices = np.loadtxt('training_split_indices.txt', dtype='int', comments='#', delimiter='\t')
validation_split_indices = np.loadtxt('validation_split_indices.txt', dtype='int', comments='#', delimiter='\t')

x_train = original_x_train[training_split_indices,:,:,:]
y_train = original_y_train[training_split_indices]

x_val = original_x_train[validation_split_indices,:,:,:]
y_val = original_y_train[validation_split_indices]

np.savez('splitted_cifar10_dataset.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

print('##### cifar10 Dataset #####')
cifar10_totals=np.zeros(4)
for i in range(10):
	num_train = len(np.where(original_y_train == i)[0])
	num_val = 0
	num_test = len(np.where(y_test == i)[0])
	total = num_train + num_val + num_test

	cifar10_totals[0] += num_train
	cifar10_totals[1] += num_val
	cifar10_totals[2] += num_test
	cifar10_totals[3] += total

	print('digit:{},\tnum_train:{},\tnum_val:{},\tnum_test:{},\ttotal:{}'.format(i,num_train, num_val, num_test, total))

print('TOTAL:,\tnum_train:{},\tnum_val:{},\tnum_test:{},\ttotal:{}'.format(cifar10_totals[0], cifar10_totals[1], cifar10_totals[2], cifar10_totals[3]))

print('##### Splitted Dataset #####')
splitted_totals=np.zeros(4)
for i in range(10):
	num_train = len(np.where(y_train == i)[0])
	num_val = len(np.where(y_val == i)[0])
	num_test = len(np.where(y_test == i)[0])
	total = num_train + num_val + num_test

	splitted_totals[0] += num_train
	splitted_totals[1] += num_val
	splitted_totals[2] += num_test
	splitted_totals[3] += total

	print('digit:{},\tnum_train:{},\tnum_val:{},\tnum_test:{},\ttotal:{}'.format(i,num_train, num_val, num_test, total))

print('TOTAL:,\tnum_train:{},\tnum_val:{},\tnum_test:{},\ttotal:{}'.format(splitted_totals[0], splitted_totals[1], splitted_totals[2], splitted_totals[3]))


