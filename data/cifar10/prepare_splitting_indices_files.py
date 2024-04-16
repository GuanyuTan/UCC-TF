import numpy as np

import keras
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:{}'.format(x_train.shape))
print('x_test shape:{}'.format(x_test.shape))

print('##### cifar10 training dataset #####')
cifar10_train_digit_dict = dict()
cifar10_num_train_list = list()
for i in range(10):
	digit_key = 'digit' + str(i)

	temp_digit_dict = dict()

	temp_digit_dict['train_indices'] = np.where(y_train == i)[0]
	temp_digit_dict['num_train'] = len(temp_digit_dict['train_indices'])

	cifar10_train_digit_dict[digit_key] = temp_digit_dict

	cifar10_num_train_list.append(temp_digit_dict['num_train'])

	print('digit:{}, num_train:{}'.format(i,temp_digit_dict['num_train']))

cifar10_num_train_arr = np.array(cifar10_num_train_list)

splitted_num_val_arr = (cifar10_num_train_arr/5).astype('int')

splitted_val_indices_list = list()
splitted_train_indices_list = list()
print('##### Splitted dataset #####')
for i in range(10):
	digit_key = 'digit' + str(i)

	cifar10_num_train = cifar10_train_digit_dict[digit_key]['num_train']
	splitted_num_val = splitted_num_val_arr[i]
	splitted_num_train = cifar10_num_train - splitted_num_val

	print('digit:{},\tcifar10_num_train:{},\tsplitted_num_train:{},\tsplitted_num_val:{}'.format(i,cifar10_num_train,splitted_num_train,splitted_num_val))

	temp_cifar10_train_indices = cifar10_train_digit_dict[digit_key]['train_indices']

	np.random.shuffle(temp_cifar10_train_indices)

	splitted_val_indices_list += list(temp_cifar10_train_indices[:splitted_num_val])
	
	splitted_train_indices_list	+= list(temp_cifar10_train_indices[splitted_num_val:])

print('splitted_val_indices_list length:{}'.format(len(splitted_val_indices_list)))
print('splitted_train_indices_list length:{}'.format(len(splitted_train_indices_list)))

splitted_val_indices_arr = (np.array(splitted_val_indices_list)).reshape((-1,1))
splitted_train_indices_arr = (np.array(splitted_train_indices_list)).reshape((-1,1))

print('splitted_val_indices_arr shape:{}'.format(splitted_val_indices_arr.shape))
print('splitted_train_indices_arr shape:{}'.format(splitted_train_indices_arr.shape))


np.savetxt('validation_split_indices.txt', splitted_val_indices_arr, fmt='%d', delimiter='\t', newline='\n', header='cifar10 training set indices chosen for validation split', footer='', comments='# ', encoding=None)
np.savetxt('training_split_indices.txt', splitted_train_indices_arr, fmt='%d', delimiter='\t', newline='\n', header='cifar10 training set indices chosen for training split', footer='', comments='# ', encoding=None)



