import json
from pathlib import Path
import sys

directory = 'images_dataset'

files = Path(directory).glob('*.json')
list_of_files = [sys.argv[1]] #list(files)

for file in list_of_files:
    with open(file) as jsonFile:
        data = json.load(jsonFile)

    for frame in data:
        if len(frame['walls']) > 4:
            new_walls = frame['walls'][::-1]
            for w in new_walls:
                w['x1'], w['y1'], w['x2'], w['y2'] = w['x2'], w['y2'], w['x1'], w['y1']

            frame['walls'] = new_walls

    with open(file, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4, sort_keys=True)
        jsonFile.close()

