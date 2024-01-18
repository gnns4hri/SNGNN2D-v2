import glob
import json
import cv2
import random
import os
import sys

if len(sys.argv) == 2:
    path = os.path.abspath(sys.argv[1])
else:
    print('You must indicate the path of the directory containing the json files')
    exit()

files = []
for path, subdirs, ifiles in os.walk(path):
    for name in ifiles:
        filename = os.path.join(path, name)
        if filename.endswith('.json'):
            files.append(filename)

splits = [('dev_set.txt', 0.1), ('test_set.txt', 0.1), ('train_set.txt', 0.8)]

random.shuffle(files)

# Some output...
N = len(files)

# Main loop. Do the splitting and prepare for data augmentation *of the training set*
for out_filename, split in splits:
    n = int(split*N)

    to_add = files[:n]
    files = files[n:]

    random.shuffle(to_add)
    print(out_filename, len(to_add))
    out_file = open(out_filename, 'w')
    for i in to_add:
        if os.path.exists(i.split('.')[0] + '__Q1.png'):
            line_text = i+'\n'
            out_file.write(line_text)
    out_file.close()




