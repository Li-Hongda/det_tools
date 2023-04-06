import os
import mmengine
from mmrotate.evaluation.metrics import DOTAMetric
import argparse
from typing import Union, List, Tuple
from mmengine.logging import MMLogger
import numpy as np


class_meta = {
    "classes": ('C', 'F', 'B', 'K', 'J', 'I', 'H', 'D', 'E', 'G', 'A')
}

def parse_args():
    parser = argparse.ArgumentParser(description='Offline evaluation code for various score_thr')
    parser.add_argument("pkl",type=str)
    parser.add_argument("--save-dir", type=str)
    # parser.add_argument("--score_thrs", nargs='+', default=[0.5])    
    parser.add_argument("--score_thrs", nargs='+', default=list(np.arange(0.1,0.9,0.1)))
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--metric", type=str, default='f1_score', choices=['mAP', 'f1_score'])
    parser.add_argument("--merge_results", default=False, action="store_true")
    args = parser.parse_args()
    return args    



def parse_pkl_result(pickle_file: str,
                     score_thr: float):
    detections = mmengine.load(pickle_file)
    results = []
    for per_image_result in mmengine.track_iter_progress(detections):
        gt_instances = per_image_result['gt_instances']
        gt_ignore_instances = per_image_result['ignored_instances']
        if gt_instances == {}:
            ann = dict()
        else:
            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
        result = dict()
        pred = per_image_result['pred_instances']
        result['img_id'] = per_image_result['img_id']
        result['scores'] = pred['scores'].cpu().numpy()
        valid_index = np.nonzero(result['scores'] > score_thr)[0]
        result['scores'] = result['scores'][valid_index]
        result['bboxes'] = pred['bboxes'].cpu().numpy()[valid_index]
        result['labels'] = pred['labels'].cpu().numpy()[valid_index]

        result['pred_bbox_scores'] = []
        for label in range(len(class_meta['classes'])):
            index = np.where(result['labels'] == label)[0]
            pred_bbox_scores = np.hstack([
                result['bboxes'][index], result['scores'][index].reshape(
                    (-1, 1))
            ])
            result['pred_bbox_scores'].append(pred_bbox_scores)
        results.append((ann, result))
    return results     
    

def offline_eval(results:List[Tuple],
                 save_dir:str,
                 epochname:str,
                 iou_thrs:Union[float, List[float]] = 0.5,
                 metric: Union[str, List[str]] = 'mAP',
                 merge_results: bool = False):
    evaluator = DOTAMetric(iou_thrs, metric=metric)
    setattr(evaluator, 'log_dir', f'{save_dir}/test_{epochname}.log')
    evaluator.results = results
    evaluator.dataset_meta = class_meta
    if merge_results:
        eval_results.outfile_prefix = os.path.join(save_dir, 'merged_results')
    eval_results = evaluator.evaluate(len(results))
    mean_f1_score, _ = list(map(lambda x: eval_results[x], eval_results))
    return mean_f1_score
    
    
def main():
    args = parse_args()
    epoch_name = args.pkl.split('/')[-1].split('.')[0]
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = ('/').join(args.pkl.split('/')[:2])
    log_dir = f'{save_dir}/test_{epoch_name}.log'
    if os.path.exists(log_dir):
        os.remove(log_dir)
    logger = MMLogger(name='Offline Evaluation',log_file=f'{save_dir}/test_{epoch_name}.log', file_mode='a')
    scores = []
    for score_thr in args.score_thrs:
        logger.info(f"Start evaluating with score_thr={score_thr}!")
        results = parse_pkl_result(args.pkl, score_thr)
        score = offline_eval(results, 
                             save_dir, 
                             epoch_name,
                             iou_thrs=args.iou_thr,
                             metric=args.metric, 
                             merge_results = args.merge_results)
        logger.info(f'The score under this threshold is {100 * score:.4f}\n')
        scores.append(score)
    sorted_id = sorted(range(len(scores)), key=lambda k: scores[k])
    sorted_scores = sorted(scores)
    logger.info(f"All the evaluation has finished!")
    logger.info(f"The best {args.metric} is {sorted_scores[-1]:.4f} under the score_thr {args.score_thrs[sorted_id[-1]]}!")
            
            
if __name__ == '__main__':
    main()
