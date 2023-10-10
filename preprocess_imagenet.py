#!/bin/python3
import os, json, pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

filenames = [('train_data_batch_%d' % (i+1)) for i in range(1)] + ['val_data']
filenames = [os.path.join('dataset_dir', 'Imagenet64', f) for f in filenames]

for filename in filenames:
	print('%s...' % filename)

	if os.path.isfile(filename):
		res = unpickle(filename)
		with open(filename + '.json', 'w') as f:
			json.dump({'data': res['data'].tolist(), 'labels': res['labels']}, f)
		print('Converted')
		# os.remove(filename)
	else:
		print('Not Found')
