import numpy as np
from keras import backend as K
from itertools import combinations


class Dataset(object):
	def __init__(self, num_instances=2, data_augment=True, num_samples_per_class=16, object_arr=None, ucc_start=1, ucc_end=10):
		
		# number of instances per bag
		self._num_instances = num_instances
		# number of bags per class in each batch
		self._num_samples_per_class = num_samples_per_class
		#  array of objects taken
		self._object_arr = object_arr

		self._ucc_start = ucc_start
		self._ucc_end = ucc_end
		self._data_augment = data_augment

		self._num_objects = len(self._object_arr)

		self._num_classes = self._ucc_end - self._ucc_start + 1

		splitted_dataset = np.load('../data/cifar10/splitted_cifar10_dataset.npz')

		x_train = splitted_dataset['x_train']
		y_train = splitted_dataset['y_train']
		x_val = splitted_dataset['x_val']
		y_val = splitted_dataset['y_val']
		# x_test = splitted_dataset['x_test']
		# y_test = splitted_dataset['y_test']

		del splitted_dataset

		# reshape to instances
		x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
		x_train = x_train.astype('float32')
		x_train /= 255
		
		x_train = (x_train-np.mean(x_train,axis=(0,1,2))[np.newaxis,np.newaxis,np.newaxis,:])/np.std(x_train,axis=(0,1,2))[np.newaxis,np.newaxis,np.newaxis,:]
		print('x_train shape:', x_train.shape)
		print(x_train.shape[0], 'train samples')

		self._x_train = x_train
		self._y_train = y_train

		del x_train
		del y_train

		x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 3)
		x_val = x_val.astype('float32')
		x_val /= 255
		x_val = (x_val-np.mean(x_val,axis=(0,1,2))[np.newaxis,np.newaxis,np.newaxis,:])/np.std(x_val,axis=(0,1,2))[np.newaxis,np.newaxis,np.newaxis,:]
		print(x_val.shape[0], 'val samples')
		
		self._x_val = x_val
		self._y_val = y_val

		del x_val
		del y_val

		self._object_dict = self.get_object_dict()
		self._class_dict_train = self.get_class_dict()
		self._class_dict_val = self.get_class_dict()

		self._labels = self.generate_labels()

	def get_object_dict(self):
		object_dict = dict()
		# for object every combination
		for i in range(self._num_objects):
			object_key = 'object' + str(i)
			object_value = self._object_arr[i]

			temp_object_dict = dict()

			temp_object_dict['value'] = object_value
			temp_object_dict['train_indices'] = np.where(self._y_train == object_value)[0]
			temp_object_dict['num_train'] = len(temp_object_dict['train_indices'])
			temp_object_dict['val_indices'] = np.where(self._y_val == object_value)[0]
			temp_object_dict['num_val'] = len(temp_object_dict['val_indices'])

			print('{}:{}, num_train:{}, num_val:{}'.format(object_key,object_value,temp_object_dict['num_train'],temp_object_dict['num_val']))

			object_dict[object_key] = temp_object_dict

		return object_dict

	def augment_image(self, image, augment_ind):
		if augment_ind == 0:
			return image
		elif augment_ind == 1:
			return np.rot90(image)
		elif augment_ind == 2:
			return np.rot90(image,2)
		elif augment_ind == 3:
			return np.rot90(image,3)
		elif augment_ind == 4:
			return np.fliplr(image)
		elif augment_ind == 5:
			return np.rot90(np.fliplr(image))
		elif augment_ind == 6:
			return np.rot90(np.fliplr(image),2)
		elif augment_ind == 7:
			return np.rot90(np.fliplr(image),3)

	def get_class_dict(self):
		# for object every combination
		elements_arr = np.arange(self._num_objects)
		class_dict = dict()
		for i in range(self._num_classes):
			class_key = 'class_' + str(i)

			temp_class_dict = dict()
			# print(elements_arr)
			elements_list = list()
			for j in combinations(elements_arr,i+self._ucc_start):
				elements_list.append(np.array(j))

			elements_array = np.array(elements_list)
			np.random.shuffle(elements_array)
			temp_class_dict['tuples_arr'] = elements_array
			temp_class_dict['num_tuples'] = len(temp_class_dict['tuples_arr'])
			temp_class_dict['index'] = 0

			# print(temp_class_dict['tuples_arr'].shape)
			# print('{}, num_tuples:{}'.format(class_key,temp_class_dict['num_tuples']))

			class_dict[class_key] = temp_class_dict

		return class_dict

	def one_hot_label(self, label):
		one_hot_label = np.zeros(self._num_classes,dtype=np.int32)
		one_hot_label[label]=1
		return one_hot_label

	def generate_labels(self):
		labels_list = list()
		for i in range(self._num_classes):
			labels_list.append(self.one_hot_label(i))

		labels_arr = np.repeat(np.array(labels_list),self._num_samples_per_class,axis=0)
		# print(labels_arr)

		return labels_arr

	def get_sample_data_train(self, indices_arr):
		# TODO Check this function
		sample = np.array(self._x_train[indices_arr,:,:,:])
		if self._data_augment:
			for index in range(len(indices_arr)):
				augment_ind = np.random.randint(8)
				sample[index] = self.augment_image(sample[index],augment_ind=augment_ind)
		return sample

	def next_batch_train(self):
		indices_list = list()
		for i in range(self._num_classes):
			class_key = 'class_' + str(i)
			# print('class_key:{}'.format(class_key))
			for j in range(self._num_samples_per_class):
				ind = self._class_dict_train[class_key]['index']
				
				temp_elements = self._class_dict_train[class_key]['tuples_arr'][ind,:]

				num_elements = temp_elements.shape[0]

				num_instances_per_element = self._num_instances // num_elements
				remainder_size = self._num_instances % num_elements

				num_instances_arr = np.repeat(num_instances_per_element,num_elements)
				num_instances_arr[:remainder_size] += 1

				for k in range(num_elements):
					object_key = 'object' + str(temp_elements[k])

					num_instances = num_instances_arr[k]

					indices_list += list(self._object_dict[object_key]['train_indices'][:num_instances])

					np.random.shuffle(self._object_dict[object_key]['train_indices'])


				self._class_dict_train[class_key]['index'] += 1

				if self._class_dict_train[class_key]['index'] >= self._class_dict_train[class_key]['num_tuples']:
					self._class_dict_train[class_key]['index'] = 0
					np.random.shuffle(self._class_dict_train[class_key]['tuples_arr'])

		
		indices_arr = np.array(indices_list)

		samples_arr = self.get_sample_data_train(indices_arr)

		samples_arr = np.reshape(samples_arr, (-1,self._num_instances,samples_arr.shape[1],samples_arr.shape[2],samples_arr.shape[3]))

		samples_data = np.transpose(samples_arr,(1,0,2,3,4))

		samples = list(samples_data)

		labels = self._labels

		return samples, [labels,samples_arr]

	def get_sample_data_val(self, indices_arr):
		sample = np.array(self._x_val[indices_arr,:,:,:])

		return sample

	def next_batch_val(self):
		indices_list = list()
		for i in range(self._num_classes):
			class_key = 'class_' + str(i)
			# print('class_key:{}'.format(class_key))
			for j in range(self._num_samples_per_class):
				ind = self._class_dict_val[class_key]['index']
				
				temp_elements = self._class_dict_val[class_key]['tuples_arr'][ind,:]

				num_elements = temp_elements.shape[0]

				num_instances_per_element = self._num_instances // num_elements
				remainder_size = self._num_instances % num_elements

				num_instances_arr = np.repeat(num_instances_per_element,num_elements)
				num_instances_arr[:remainder_size] += 1

				for k in range(num_elements):
					object_key = 'object' + str(temp_elements[k])

					num_instances = num_instances_arr[k]

					indices_list += list(self._object_dict[object_key]['val_indices'][:num_instances])

					np.random.shuffle(self._object_dict[object_key]['val_indices'])


				self._class_dict_val[class_key]['index'] += 1

				if self._class_dict_val[class_key]['index'] >= self._class_dict_val[class_key]['num_tuples']:
					self._class_dict_val[class_key]['index'] = 0
					np.random.shuffle(self._class_dict_val[class_key]['tuples_arr'])

		
		indices_arr = np.array(indices_list)

		samples_arr = self.get_sample_data_val(indices_arr)

		samples_arr = np.reshape(samples_arr, (-1,self._num_instances,samples_arr.shape[1],samples_arr.shape[2],samples_arr.shape[3]))

		samples_data = np.transpose(samples_arr,(1,0,2,3,4))

		samples = list(samples_data)

		labels = self._labels

		return samples, [labels,samples_arr]

class DatasetTest(object):
	def __init__(self, num_instances=2, num_samples_per_class=16, object_arr=None, ucc_start=1, ucc_end=10):
		
		self._num_instances = num_instances
		self._num_samples_per_class = num_samples_per_class
		self._object_arr = object_arr
		self._ucc_start = ucc_start
		self._ucc_end = ucc_end

		self._num_objects = len(self._object_arr)

		self._num_classes = self._ucc_end - self._ucc_start + 1

		splitted_dataset = np.load('../data/cifar10/splitted_cifar10_dataset.npz')

		x_test = splitted_dataset['x_test']
		y_test = splitted_dataset['y_test']

		del splitted_dataset

		x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
		x_test = x_test.astype('float32')
		x_test /= 255
		x_test = (x_test-np.mean(x_test, axis=(0,1,2))[:,np.newaxis,np.newaxis,np.newaxis])/np.std(x_test,axis=(0,1,2))[:,np.newaxis,np.newaxis,np.newaxis]

		print('x_test shape:', x_test.shape)
		print(x_test.shape[0], 'test samples')
		print('y_test shape:', y_test.shape)

		self._x_test = x_test
		self._y_test = y_test

		self._object_dict = self.get_object_dict()
		self._class_dict_test = self.get_class_dict()

		self._labels = self.generate_labels()

	def get_object_dict(self):
		object_dict = dict()
		for i in range(self._num_objects):
			object_key = 'object' + str(i)
			object_value = self._object_arr[i]

			temp_object_dict = dict()

			temp_object_dict['value'] = object_value
			temp_object_dict['test_indices'] = np.where(self._y_test == object_value)[0]
			temp_object_dict['num_test'] = len(temp_object_dict['test_indices'])

			print('{}:{}, num_test:{}'.format(object_key,object_value,temp_object_dict['num_test']))

			object_dict[object_key] = temp_object_dict

		return object_dict


	def get_class_dict(self):
		elements_arr = np.arange(self._num_objects)
		class_dict = dict()
		for i in range(self._num_classes):
			class_key = 'class_' + str(i)

			temp_class_dict = dict()
			# print(elements_arr)
			elements_list = list()
			for j in combinations(elements_arr,i+self._ucc_start):
				elements_list.append(np.array(j))

			elements_array = np.array(elements_list)

			temp_class_dict['tuples_arr'] = elements_array
			temp_class_dict['num_tuples'] = len(temp_class_dict['tuples_arr'])
			temp_class_dict['index'] = 0

			# print(temp_class_dict['tuples_arr'].shape)
			# print('{}, num_tuples:{}'.format(class_key,temp_class_dict['num_tuples']))

			class_dict[class_key] = temp_class_dict

		return class_dict

	def one_hot_label(self, label):
		one_hot_label = np.zeros(self._num_classes,dtype=np.int)
		one_hot_label[label]=1
		return one_hot_label

	def generate_labels(self):
		labels_list = list()
		for i in range(self._num_classes):
			labels_list.append(self.one_hot_label(i))

		labels_arr = np.repeat(np.array(labels_list),self._num_samples_per_class,axis=0)
		# print(labels_arr)

		return labels_arr

	def get_sample_data_test(self, indices_arr):
		sample = np.array(self._x_test[indices_arr,:,:,:])
		# print('Sample shape:{}'.format(sample.shape))
		return sample

	def next_batch_test(self):
		indices_list = list()
		for i in range(self._num_classes):
			class_key = 'class_' + str(i)
			# print('class_key:{}'.format(class_key))
			for j in range(self._num_samples_per_class):
				ind = self._class_dict_test[class_key]['index']
				
				temp_elements = self._class_dict_test[class_key]['tuples_arr'][ind,:]

				num_elements = temp_elements.shape[0]

				num_instances_per_element = self._num_instances // num_elements
				remainder_size = self._num_instances % num_elements

				num_instances_arr = np.repeat(num_instances_per_element,num_elements)
				num_instances_arr[:remainder_size] += 1

				for k in range(num_elements):
					object_key = 'object' + str(temp_elements[k])

					num_instances = num_instances_arr[k]

					indices_list += list(self._object_dict[object_key]['test_indices'][:num_instances])

					np.random.shuffle(self._object_dict[object_key]['test_indices'])


				self._class_dict_test[class_key]['index'] += 1

				if self._class_dict_test[class_key]['index'] >= self._class_dict_test[class_key]['num_tuples']:
					self._class_dict_test[class_key]['index'] = 0

		
		indices_arr = np.array(indices_list)

		samples_arr = self.get_sample_data_test(indices_arr)

		samples_arr = np.reshape(samples_arr, (-1,self._num_instances,samples_arr.shape[1],samples_arr.shape[2],samples_arr.shape[3]))

		samples_arr = np.transpose(samples_arr,(1,0,2,3,4))

		samples = list(samples_arr)

		labels = self._labels

		return samples, labels

