import os
import copy
import argparse
import mmengine
import numpy as np
import torch
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

def load_results_single(srcPath, ignored_classes=[]):
    pickle_results = mmengine.load(srcPath)
    per_image_results = dict()
    for per_image_dets in mmengine.track_iter_progress(pickle_results):
        pred = per_image_dets['pred_instances']
        imgID = per_image_dets['img_id']
        scores = pred['scores'].numpy()
        bboxes = pred['bboxes'].numpy()
        labels = pred['labels'].numpy()
        if scores.shape == (0,):
            continue
        dets = np.concatenate((bboxes, scores[:, None]), axis=1)
        for idx, classID in enumerate(list(labels)):
            if imgID in per_image_results:
                if classID in per_image_results[imgID]:
                    per_image_results[imgID][classID] = np.concatenate((per_image_results[imgID][classID], dets[idx][None]), axis=0)
                else:
                    per_image_results[imgID][classID] = dets[idx][None]
            else:
                per_image_results[imgID] = {classID: dets[idx][None]}        
    return per_image_results, pickle_results

def load_results_single1(srcPath, ignored_classes=[]):
    pickle_results = mmengine.load(srcPath)
    per_image_results = dict()
    for per_image_dets in mmengine.track_iter_progress(pickle_results):
        pred = per_image_dets['pred_instances']
        imgID = per_image_dets['img_id']
        scores = pred['scores'].numpy()
        scores = np.where(scores > 0.45, scores + 0.3, scores)
        bboxes = pred['bboxes'].numpy()
        labels = pred['labels'].numpy()
        if scores.shape == (0,):
            continue
        dets = np.concatenate((bboxes, scores[:, None]), axis=1)
        for idx, classID in enumerate(list(labels)):
            if imgID in per_image_results:
                if classID in per_image_results[imgID]:
                    per_image_results[imgID][classID] = np.concatenate((per_image_results[imgID][classID], dets[idx][None]), axis=0)
                else:
                    per_image_results[imgID][classID] = dets[idx][None]
            else:
                per_image_results[imgID] = {classID: dets[idx][None]}        
    return per_image_results, pickle_results
    
def load_results(args, logger):
    logger.info("Starting loading model detection results")
    for i in range(len(args.srcPaths)):
        # detections, pickle_results = load_results_single(args.srcPaths[i], args.ignored_classes)    
        if i == 0:
            detections, pickle_results = load_results_single(args.srcPaths[i], args.ignored_classes)
            dets = copy.deepcopy(detections)
            continue
        else:
            detections, pickle_results = load_results_single1(args.srcPaths[i], args.ignored_classes)
        for imgID, results in detections.items():
            # detections, pickle_results = load_results_single1(args.srcPaths[i], args.ignored_classes)
            if imgID in dets:
                for classID, result in results.items():
                    if classID in dets[imgID]:
                        dets[imgID][classID] = np.concatenate((dets[imgID][classID], result), axis=0)
                    else:
                        dets[imgID][classID] = result
            else:
                dets[imgID] = results
    logger.info("Loading results finished!!")
    return dets, pickle_results  
                

def save_results(args, logger, detections, pickle_results):
    for result in pickle_results:
        imgID = result['img_id']
        pred = result['pred_instances']
        if imgID in detections:
            dets = detections[imgID]
            bboxes, scores, labels = [[] for _ in range(3)]
            for k, v in dets.items():
                num = len(v)
                bboxes.append(v[:,:5])
                scores.append(v[:,5])
                for i in range(num):
                    labels.append(k)
            per_img_bboxes = torch.from_numpy(np.concatenate(bboxes))
            per_img_scores = torch.from_numpy(np.concatenate(scores))
            per_img_labels = torch.from_numpy(np.hstack(labels))
            pred['bboxes'] = per_img_bboxes
            pred['scores'] = per_img_scores
            pred['labels'] = per_img_labels
        else:
            continue
    save_dir = os.path.join(args.save_dir, 'merged_result.pkl')
    mmengine.dump(pickle_results, save_dir)
    
    

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
    detections, pickple_results = load_results(args, logger)
    merged_detections = result_merge(args, logger, detections)
    logger.info("Starting saving merged results!")
    save_results(args, logger, merged_detections, pickple_results)
    
    

if __name__ == '__main__':
    main()
