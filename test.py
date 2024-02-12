from cProfile import label
import cv2

from utils.socnav2d_V2_API import *
from dataset.socnav2d_dataset import *
import argparse
import time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test SNGNN2D-v2 model from a json file or from a txt file with containing a list of json files')
    parser.add_argument('--file', '-f', type=str, required=True, help='Specify the path to the JSON or txt file to test')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Specify the path to the model parameters')
    parser.add_argument('--cuda', '-c', action='store_true', help='Use GPU if available')
    parser.add_argument('--scenarios_path', '-s', type=str, required=False, default='./videos/',help='Specify the path to videos of the scenarios')
    args = parser.parse_args()

    assert args.file.endswith('.json') or args.file.endswith('.txt'), "The file must be a json or a txt file"
    assert os.path.exists(args.file), "The test file does not exist"
    assert os.path.exists(args.model_path), "The path to the model does not exist"

    scenario_path = args.scenarios_path
    if not scenario_path.endswith('/'):
        scenario_path += '/'

    if args.file.endswith('.json'):
        filenames = [args.file]
    else:
        filenames = open(args.file, 'r').read().splitlines()

    device = 'cpu'
    if args.cuda:
        device = 'cuda'

    sngnn = SocNavAPI(base=args.model_path + "/", device=device)

    for f in filenames:

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
        ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)    

        label_filename = '.'.join(f.split('.')[:-1]) + '__Q1.png'
        label = cv2.imread(label_filename)
        if label is None:
            print('Couldn\'t read label file', label_filename)
            black_img = np.zeros((300, 300,3), dtype=np.uint8)
            image = np.concatenate((ret, black_img), axis=1)
        else:
            label = cv2.resize(label, (300, 300), interpolation=cv2.INTER_CUBIC)
            image = np.concatenate((ret, label), axis=1)
        scenario_filename = scenario_path+f.split('/')[-1].split('.')[0] + '.mp4' 
        video = cv2.VideoCapture(scenario_filename)
        if video.isOpened():
            new_frame = True
            while new_frame:
                new_frame, frame = video.read()
                if new_frame:
                    real_img = frame
            video.release()
            real_img = cv2.resize(real_img, (300, 300), interpolation=cv2.INTER_CUBIC)
            black_img = np.zeros((300, 150,3), dtype=np.uint8)
            real_img = np.concatenate((black_img, real_img), axis=1)
            real_img = np.concatenate((real_img, black_img), axis=1)
            image = np.concatenate((real_img,image), axis=0)

        first_time = True

        while True:
            if first_time:
                try:
                    cv2.imshow("SNGNN2D-v2 output - Ground truth", image)
                    k = cv2.waitKey(1)
                    if k == 27: # or cv2.getWindowProperty("SNGNN2D-v2 output - Ground truth", cv2.WND_PROP_VISIBLE) < 1:
                        cv2.destroyAllWindows()
                        sys.exit(0)
                    else:
                        if k == 13:
                            break
                except cv2.error as e:

                        directory = 'images_test'
                        print(f'It is not possible to display the image, saving them into {directory} instead')
                        first_time = False

                        try:
                            os.makedirs(directory, exist_ok=True)
                        except OSError as error:
                            print('Exception creating directory:', directory, f'bcause of error {error}')
                            sys.exit(1)
            else:
                filename = f.split('/')[-1].split('.')[0] + '.png'
                cv2.imwrite(f'{directory}/SNGNN2D-v2 output-{filename}', image)
                break
    



