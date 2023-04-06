import os
import cv2
import glob
import mmengine
from loguru import logger
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='visualize')
    parser.add_argument("--src_img_dir",default='')
    parser.add_argument("--src_lab_dir",default='')
    parser.add_argument("--dst_dir",default='')
    parser.add_argument("--type", choices=['gt', 'det'], default='gt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # assert args.srcPath is not None,f"{args.srcPath} must be specific."
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)
    # else:
    #     logger.info(f"{args.dstPath} already exists.Do you want to overwrite it?(y/n)")
    #     if input().lower() == "y":
    #         os.makedirs(args.dstPath, exist_ok=True)
    logger.info("Starting collecting data infos.")
    img_dir = os.path.join(args.src_img_dir,'images')
    label_dir = args.src_lab_dir
    # labels = glob.glob(os.path.join(label_dir,'*.txt'))
    file_generator = mmengine.scandir(label_dir)
    for file in mmengine.track_iter_progress(list(file_generator)):
        img = cv2.imread(os.path.join(img_dir,file.replace('.txt', '.png')))
        with open(os.path.join(label_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                for line in lines:
                    line = line.strip().split(" ")
                    class_name = line[-2]
                    score = float(line[-1])
                    box = list(map(float, line[0:8]))
                    box = np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [box], True, [0, 255, 0], thickness=3)
                    cv2.putText(img, f"{class_name }|{score:.2f}",(box[0][0][0], box[0][0][1]-20), \
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale =0.5 ,thickness= 2, color=(0,255,0))
        cv2.imwrite(os.path.join(args.dst_dir,file.replace('.txt', '.jpg')), img)
    logger.info("Visualizing gt bboxes finish!!")
    

    
if __name__ == '__main__':
    main()
