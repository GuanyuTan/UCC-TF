import numpy as np

splitted_dataset = np.load('splitted_mnist_dataset.npz')

x_train = splitted_dataset['x_train']
y_train = splitted_dataset['y_train']
x_val = splitted_dataset['x_val']
y_val = splitted_dataset['y_val']
x_test = splitted_dataset['x_test']
y_test = splitted_dataset['y_test']

print(x_train.dtype)
print(y_train.dtype)
print(x_val.dtype)
print(y_val.dtype)
print(x_test.dtype)
print(y_test.dtype)

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

