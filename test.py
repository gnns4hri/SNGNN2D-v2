from utils.socnav2d_V2_API import *
from dataset.socnav2d_dataset import *
import argparse
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test SNGNN2D-v2 model from a json file or from a txt file with containing a list of json files')
    parser.add_argument('--file', '-f', type=str, required=True, help='Specify the path to the JSON or txt file to test')
    parser.add_argument('--path', '-p', type=str, required=True, help='Specify the path to the model parameters')
    parser.add_argument('--cuda', '-c', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    assert args.file.endswith('.json') or args.file.endswith('.txt'), "The json file must be a json or a txt file"
    assert os.path.exists(args.file), "The test file does not exist"
    assert os.path.exists(args.path), "The path to the model does not exist"

    if args.file.endswith('.json'):
        filenames = [args.file]
    else:
        filenames = open(args.file, 'r').read().splitlines()

    device = 'cpu'
    if args.cuda:
        device = 'cuda'

    sngnn = SocNavAPI(base=args.path + "/", device=device)

    for f in filenames:
        print(f)

        if not os.path.exists(f):
            continue

        with open(f) as json_file:
            data = json.load(json_file)

        data.reverse()

        time_0 = time.time()
        graph = SocNavDataset(data, net=sngnn.net, mode='test', raw_dir='', alt='8', debug=True, device=device)

        time_1 = time.time()
        ret = sngnn.predictOneGraph(graph)[0]

        time_2 = time.time()
        print("total time", time_2 - time_0, "graph time", time_1 - time_0, "inference time", time_2 - time_1)

        ret = ret.reshape(image_width, image_width)

        ret = ret.cpu().detach().numpy()
        ret = (255.*(ret+1)/2.).astype(np.uint8)
        ret = cv2.resize(ret,  (300, 300), interpolation=cv2.INTER_CUBIC)

        label_filename = f.split('.')[0] + '__Q1.png'
        label = cv2.imread(label_filename)
        if label is None:
            print('Couldn\'t read label file', label_filename)
            image = ret
        else:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            label = cv2.resize(label, (300, 300), interpolation=cv2.INTER_CUBIC)
            image = np.concatenate((ret, label), axis=1)

        while True:
            cv2.imshow("SNGNN2D-v2 output - Ground truth", image)
            k = cv2.waitKey(1)
            if k == 27 or cv2.getWindowProperty("SNGNN2D-v2 output - Ground truth", cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                sys.exit(0)
            else:
                if k == 13:
                    break



