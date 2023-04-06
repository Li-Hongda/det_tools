import os
import glob
import tqdm

label_path = ''

labels = glob.glob(os.path.join(label_path, '*.txt'))
classes = []
for label in tqdm.tqdm(labels):
    with open(label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line in ['imagesource:GoogleEarth\n', 'gsd:null\n']:
                continue
            splitline = line.strip().split(' ')
            cat = splitline[-2]
            if cat not in classes:
                classes.append(cat)
            else:
                continue
print(tuple(sorted(classes)))
            
            