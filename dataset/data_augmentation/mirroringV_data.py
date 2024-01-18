import json
import os
import sys
import math
import cv2

if len(sys.argv)<2:
    print("You must specify a json/txt file")
    exit()

if sys.argv[1].endswith('.json'):
	filenames = [sys.argv[1]]
	newFileList = None
else:
	filenames = open(sys.argv[1], 'r').read().splitlines()
	newFileList = []

for filename in filenames:
	print(filename)

	if not filename.endswith('.json'):
		continue

	if not os.path.exists(filename):
		continue

	file_path, json_file = os.path.split(filename)
	json_save = file_path + '/mV_'+ json_file
	if 'mV_' in filename:
		continue

	# Read JSON data into the datastore variable
	if filename:
		with open(filename, 'r') as f:
			datastore = json.load(f)
			f.close()

	for data in datastore:
		data['command'][2] = -data['command'][2]

		for i in range(len(data['goal'])):
			data['goal'][i]['x'] = -data['goal'][i]['x']

		for i in range(len(data['objects'])):
			data['objects'][i]['a'] = math.atan2(math.sin(data['objects'][i]['a']), -math.cos(data['objects'][i]['a']))

			data['objects'][i]['x'] = -data['objects'][i]['x']
			data['objects'][i]['vx'] = -data['objects'][i]['vx']
			data['objects'][i]['va'] = -data['objects'][i]['va']

		for i in range(len(data['people'])):
			data['people'][i]['a'] = math.atan2(math.sin(data['people'][i]['a']), -math.cos(data['people'][i]['a']))


			data['people'][i]['x'] = -data['people'][i]['x']
			data['people'][i]['vx'] = -data['people'][i]['vx']
			data['people'][i]['va'] = -data['people'][i]['va']

		for i in range(len(data['walls'])):
			data['walls'][i]['x1'], data['walls'][i]['x2'] = -data['walls'][i]['x2'], -data['walls'][i]['x1']
			data['walls'][i]['y1'], data['walls'][i]['y2'] = data['walls'][i]['y2'], data['walls'][i]['y1']
			# data['walls'][i]['x2'] = -data['walls'][i]['x2']


	label_filename = filename.split('.')[0] + '__Q1.png'
	label = cv2.imread(label_filename)
	if label is None:
		print('Couldn\'t read label file', label_filename)
		continue
	
	flipVertical = cv2.flip(label, 1)
	label_path, label_file = os.path.split(label_filename)
	image_save = label_path + '/mV_'+ label_file

	if newFileList is not None:
		newFileList.append(filename)
		newFileList.append(json_save)


	with open(json_save, 'w') as outfile: 
		json.dump(datastore, outfile, indent=4, sort_keys=True) 
		outfile.close()

	cv2.imwrite(image_save, flipVertical)

if newFileList is not None:
	out_filename = sys.argv[1].split('.')[-2] + '_DAmV.' + sys.argv[1].split('.')[-1]
	out_file = open(out_filename, 'w')
	out_file.write('\n'.join(newFileList))
	out_file.close()
