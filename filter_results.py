import os
import argparse
import numpy as np
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Offline evaluation code for various score_thr')
    parser.add_argument("--src_dir", type=str, default='')
    parser.add_argument("--dst_dir", type=str, default='')
    parser.add_argument("--score_thr", type=float, default=0.8)
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
                

if __name__ == '__main__' :
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    filter_results(src_dir, dst_dir)
            
