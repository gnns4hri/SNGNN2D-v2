import sys
import random
import json
import numpy as np
import os
import cv2
from socnav2d_V2_API import *
from socnav2d import *
import time
import math

def toColour(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def beautify_grey_image(img):
    return beautify_image(toColour(img))

def beautify_image(img):
    def convert(value):
        colors = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]]
        v = (255 - value) / 255
        if v >= 1:
            idx1 = 3
            idx2 = 3
            fract = 0
        else:
            v = v * 3
            idx1 = math.floor(v)
            idx2 = idx1 + 1
            fract = v - idx1
        r = (colors[idx2][0] - colors[idx1][0]) * fract + colors[idx1][0]
        g = (colors[idx2][1] - colors[idx1][1]) * fract + colors[idx1][1]
        b = (colors[idx2][2] - colors[idx1][2]) * fract + colors[idx1][2]
        red = r * 255
        green = g * 255
        blue = b * 255
        return red, green, blue
    for row in range(0,img.shape[0]):
        for col in range(0, img.shape[1]):
            v = float(img[row, col, 2])
            bad = 255.-v
            red, blue, green = convert(v)
            th = 215.
            img[row, col, 0] = blue
            img[row, col, 1] = green
            img[row, col, 2] = red
    return img

def test_sn(sngnn, scenario):
    ret = sngnn.predict(scenario)/255
    ret = ret.reshape(socnavImg.output_width, socnavImg.output_width)
    ret = cv2.resize(ret, (100, 100), interpolation=cv2.INTER_NEAREST)
    return ret

def test_json(sngnn, filename, line):
    ret = sngnn.predict(filename, line)
    ret = ret.reshape(socnavImg.output_width, socnavImg.output_width)
    return ret


def add_walls_to_grid(image, read_structure):
    # Get the wall points and repeat the first one to close the loop
    walls = read_structure['room']
    walls.append(walls[0])
    # Initialise data
    SOCNAV_AREA_WIDTH = 800.
    grid_rows = image.shape[0]
    grid_cols = image.shape[1]
    # Function doing hte mapping
    def coords_to_ij(y, x):
        px = int((x*grid_cols)/SOCNAV_AREA_WIDTH + grid_cols/2)
        py = int((y*grid_rows)/SOCNAV_AREA_WIDTH + grid_rows/2)
        return px, py
    for idx, w in enumerate(walls[:-1]):
        p1x, p1y = coords_to_ij(w[0], w[1])
        p2x, p2y = coords_to_ij(walls[idx+1][0], walls[idx+1][1])
        cv2.line(image, (p1y, p1x), (p2y, p2x), (50, 50, 50), 2)
        # line(grid_gray, p1, p2, 0, 15, LINE_AA);
    return image


if len(sys.argv)<2:
    print("You must specify a json/txt file")
    exit()

if sys.argv[1].endswith('.json'):
    filenames = [sys.argv[1]]
else:
    filenames = open(sys.argv[1], 'r').read().splitlines()

device = 'cpu'
if 'cuda' in sys.argv:
    device = 'cuda'

sngnn = SocNavAPI(base = './model_params/pix2pix', device = device)

for f in filenames:
    print(f)
    
    if not os.path.exists(f):
        continue

    with open(f) as json_file:
        data = json.load(json_file)

    data.reverse()


    time_0 = time.time()
    graph = SocNavDataset(data, mode='test', raw_dir='', alt='8', debug=True, device = device)

    time_1  = time.time()
    ret = sngnn.predictOneGraph(graph)[0]

    time_2 = time.time()
    print("total time", time_2 - time_0, "graph time", time_1 - time_0, "inference time", time_2 - time_1)

    ret = ret.reshape(image_width, image_width)


    # ret = np.clip(ret.cpu().detach().numpy(), 0., 1.)
    ret = ret.cpu().detach().numpy()
    ret = (255.*(ret+1)/2.).astype(np.uint8)
    ret  = cv2.resize(ret,  (300, 300), interpolation=cv2.INTER_CUBIC)

    label_filename = f.split('.')[0] + '__Q1.png'
    label = cv2.imread(label_filename)
    if label is None:
        print('Couldn\'t read label file', label_filename)
        image = ret
    else:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)    
        label = cv2.resize(label, (300, 300), interpolation=cv2.INTER_CUBIC)
        image = np.concatenate((ret, label), axis=1)
    pix2pix_filename = './images_pix2pix/'+f.split('/')[-1].split('.')[0] + '_fake_B.png'
    pix2pix = cv2.imread(pix2pix_filename)
    if pix2pix is not None:
        pix2pix = cv2.cvtColor(pix2pix, cv2.COLOR_BGR2GRAY)    
        pix2pix = cv2.resize(pix2pix, (300, 300), interpolation=cv2.INTER_CUBIC)
        image = np.concatenate((image, pix2pix), axis=1)
    real_filename = './images_pix2pix/'+f.split('/')[-1].split('.')[0] + '_real_A.png'        
    real_img = cv2.imread(real_filename)
    if real_img is not None:
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)    
        real_img = cv2.resize(real_img, (300, 300), interpolation=cv2.INTER_CUBIC)
        black_img = np.zeros((300, 300), dtype=np.uint8)
        real_img = np.concatenate((black_img, real_img), axis=1)
        real_img = np.concatenate((real_img, black_img), axis=1)
        image = np.concatenate((real_img,image), axis=0)
    

    while True:
        cv2.imshow("SNGNN2D", image)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit()
        else:
            if k == 13:
            # if time.time()-time_show > 4.:
                break



