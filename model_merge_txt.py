import os
import copy
import argparse
import mmengine
import numpy as np
import torch
import glob
from tqdm import tqdm
from mmcv.ops import nms_rotated
from mmengine.logging import MMLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Offline evaluation code for various score_thr')
    parser.add_argument("srcPaths",nargs='+', type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--ignored_classes", type=str, nargs='+',default=[])
    args = parser.parse_args()
    return args    


def collect_results(results_dir, class_ignore=[]):
    results_per_image = {}
    for result_file in glob.glob(os.path.join(results_dir, "*.txt")):
        image_id = os.path.splitext(os.path.basename(result_file))[0]
        with open(result_file, "r") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            line = line.strip().split(" ")
            class_name = line[0]
            if class_name in class_ignore:
                continue
            score = line[1]
            bbox = line[2:]
            bbox = list(map(float, bbox))
            score = float(score)
            bbox.append(score)
            bbox = np.asarray(bbox, dtype=np.float32)
            if class_name in info:
                info[class_name] = np.vstack((info[class_name], bbox[None]))
            else:
                info[class_name] = bbox[None]
        results_per_image[image_id] = info

    return results_per_image
    
def load_results(args, logger):
    results_collect = {}
    for src_results_dir in tqdm(args.srcPaths):
        src_results_collect = collect_results(src_results_dir)
        for image_id, results in src_results_collect.items():
            if image_id in results_collect:
                for class_name, result in results.items():
                    if class_name in results_collect[image_id]:
                        results_collect[image_id][class_name] \
                            = np.concatenate((results_collect[image_id][class_name], result), axis=0)
                    else:
                        results_collect[image_id][class_name] = result
            else:
                results_collect[image_id] = results
    return results_collect
                

def save_results(args, logger, detections):
    for imgID, per_img_dets in detections.items():
        lines = ""
        for classname, per_cls_det in per_img_dets.items():
            bboxes = per_cls_det[:,:8].tolist()
            scores = per_cls_det[:,8].tolist()
            for idx, (bbox, score) in enumerate(zip(bboxes, scores)):
                line = [classname, f"{score:.3f}"]
                def fstring(s):
                    return f"{s:.4f}"
                bbox = list(map(fstring, bbox))
                line.extend(bbox)
                line =f"{idx}" + " " +  " ".join(line) + "\n"
                lines += line 
            # if lines[-1] == "\n":
            #     lines = lines[:-1]
        with open(os.path.join(args.save_dir, imgID + '.txt'), "w") as f:
            f.write(lines)
    
    

def result_merge(args, logger, detections):  
    logger.info("Starting merging detections by rotated-NMS !")
    merged_detections = copy.deepcopy(detections)
    for imageId, results in mmengine.track_iter_progress(detections.items()):
        for classID, result in results.items():
            bboxes = []
            for box in result:
                bboxes.append(box[:5])
            _, keep = nms_rotated(torch.from_numpy(np.array(bboxes)), torch.from_numpy(result[:,5]),iou_threshold=0.01)
            result = result[keep.numpy()]
            merged_detections[imageId][classID] = result    

    return merged_detections

def main():
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = MMLogger(name='Model Evaluation')
    # detections, pickple_results = load_results(args, logger)
    detections = load_results(args, logger)
    merged_detections = result_merge(args, logger, detections)
    logger.info("Starting saving merged results!")
    save_results(args, logger, merged_detections)
    
    

if __name__ == '__main__':
    main()
