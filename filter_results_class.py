"""
Filter results by various score_thr for DOTA-format submissions.
e.g.:
Task
   |——————plane.txt
   └——————baseball-diamond.txt
   |—————— ...
   └—————— harbor.txt 
Each txt file is marked as follows:
    imgname score x1 y1 x2 y1 x2 y2 x1 y2   
"""

import os
import argparse
import numpy as np
import glob

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter')

def parse_args():
    parser = argparse.ArgumentParser(description='Filter results for DOTA-format submissions.')
    parser.add_argument("--src_dir", type=str, default='')
    parser.add_argument("--dst_dir", type=str, default='')
    parser.add_argument("--score_thr", default=0.3)
    args = parser.parse_args()
    return args   


def filter_results(src_dir, dst_dir):
    results = glob.glob(os.path.join(src_dir, '*.txt'))
    for result in results:
        with open(result,'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                splitline = line.strip().split(' ')
                score = splitline[1]
                if float(score) > args.score_thr:
                    new_lines.append(line)
            with open(os.path.join(dst_dir,os.path.basename(result)),'w') as fout:
                for new_line in new_lines:
                    fout.write(new_line)

def cal_scores(src_dir):
    scores = [[] for _ in range(len(CLASSES))]
    results = glob.glob(os.path.join(src_dir, '*.txt'))
    for result in results:
        with open(result,'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                splitline = line.strip().split(' ')
                category = CLASSES.index(os.path.basename(result)[6:-4])
                scores[category].append(float(splitline[1]))
    scores = np.asarray([np.vstack(scores[i]).mean() for i in range(len(CLASSES))])
    return scores

def filter_results_by_class(src_dir, dst_dir):
    score_thrs = cal_scores(src_dir)
    results = glob.glob(os.path.join(src_dir, '*.txt'))
    for result in results:
        with open(result,'r') as f:
            lines = f.readlines()
            new_lines = []
            for line in lines:
                splitline = line.strip().split(' ')
                category = CLASSES.index(os.path.basename(result)[6:-4])
                score = splitline[1]
                if float(score) > score_thrs[category]:
                    new_lines.append(line)
            with open(os.path.join(dst_dir,os.path.basename(result)),'w') as fout:
                for new_line in new_lines:
                    fout.write(new_line)
                

if __name__ == '__main__' :
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # filter_results(src_dir, dst_dir)
    filter_results_by_class(src_dir, dst_dir)
            
